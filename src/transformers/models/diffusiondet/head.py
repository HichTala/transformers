import copy
import math
from dataclasses import astuple

import torch
from torch import nn
from torch.nn.modules.transformer import _get_activation_fn
from torchvision.ops import RoIAlign

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

def convert_boxes_to_pooler_format(bboxes):
    bs, num_proposals = bboxes.shape[:2]
    sizes = torch.full((bs,), num_proposals).to(bboxes.device)
    aggregated_bboxes = bboxes.view(bs * num_proposals, -1)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=aggregated_bboxes.dtype, device=aggregated_bboxes.device), sizes
    )
    return torch.cat([indices[:, None], aggregated_bboxes], dim=1)


def assign_boxes_to_levels(
        bboxes,
        min_level,
        max_level,
        canonical_box_size,
        canonical_level,
):
    aggregated_bboxes = bboxes.view(bboxes.shape[0] * bboxes.shape[1], -1)
    area = (aggregated_bboxes[:, 2] - aggregated_bboxes[:, 0]) * (aggregated_bboxes[:, 3] - aggregated_bboxes[:, 1])
    box_sizes = torch.sqrt(area)
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8))
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class HeadDynamicK(nn.Module):
    def __init__(self, config, roi_input_shape):
        super().__init__()
        num_classes = config.num_labels

        ddet_head = DiffusionDetHead(config, roi_input_shape, num_classes)
        self.num_head = config.num_heads
        self.head_series = nn.ModuleList([copy.deepcopy(ddet_head) for _ in range(self.num_head)])
        self.return_intermediate = config.deep_supervision

        # Gaussian random feature embedding layer for time
        self.hidden_dim = config.hidden_dim
        time_dim = self.hidden_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_dim),
            nn.Linear(self.hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = config.use_focal
        self.use_fed_loss = config.use_fed_loss
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = config.prior_prob
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)


    def forward(self, features, bboxes, t):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])

        class_logits, pred_bboxes = None, None
        for head_idx, ddet_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = ddet_head(features, bboxes, time)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class DynamicConv(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.dim_dynamic = config.dim_dynamic
        self.num_dynamic = config.num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = config.pooler_resolution
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)


    def forward(self, pro_features, roi_features):
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class DiffusionDetHead(nn.Module):
    def __init__(self, config, roi_input_shape, num_classes):
        super().__init__()

        dim_feedforward = config.dim_feedforward
        nhead = config.num_attn_heads
        dropout = config.dropout
        activation = config.activation
        in_features = config.roi_head_in_features
        pooler_resolution = config.pooler_resolution
        pooler_scales = tuple(1.0 / roi_input_shape[k]['stride'] for k in in_features)
        sampling_ratio = config.sampling_ratio

        self.hidden_dim = config.hidden_dim

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
        )

        # dynamic.
        self.self_attn = nn.MultiheadAttention(self.hidden_dim, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(config)

        self.linear1 = nn.Linear(self.hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.hidden_dim)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2))

        # cls.
        num_cls = config.num_cls
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(self.hidden_dim, self.hidden_dim, False))
            cls_module.append(nn.LayerNorm(self.hidden_dim))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = config.num_reg
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(self.hidden_dim, self.hidden_dim, False))
            reg_module.append(nn.LayerNorm(self.hidden_dim))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = config.use_focal
        self.use_fed_loss = config.use_fed_loss
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(self.hidden_dim, num_classes)
        else:
            self.class_logits = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bboxes_delta = nn.Linear(self.hidden_dim, 4)
        self.scale_clamp = _DEFAULT_SCALE_CLAMP
        self.bbox_weights = (2.0, 2.0, 1.0, 1.0)

    def forward(self, features, bboxes, time_emb):
        bs, num_proposals = bboxes.shape[:2]

        # roi_feature.
        roi_features = self.pooler(features, bboxes)

        pro_features = roi_features.view(bs, num_proposals, self.hidden_dim, -1).mean(-1)

        roi_features = roi_features.view(bs * num_proposals, self.hidden_dim, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(bs, num_proposals, self.hidden_dim).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(num_proposals, bs, self.hidden_dim).permute(1, 0, 2).reshape(1, bs * num_proposals,
                                                                                      self.hidden_dim)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(bs * num_proposals, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_proposals, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(bs, num_proposals, -1), pred_bboxes.view(bs, num_proposals, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            canonical_box_size=224,
            canonical_level=4,
    ):
        super().__init__()

        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2 and isinstance(output_size[0], int) and isinstance(output_size[1], int)
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level))
        assert (len(scales) == max_level - min_level + 1)
        assert 0 <= min_level <= max_level
        assert canonical_box_size > 0

        self.output_size = output_size
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self.canonical_level = canonical_level
        self.canonical_box_size = canonical_box_size
        self.level_poolers = nn.ModuleList(
            RoIAlign(
                output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
            )
            for scale in scales
        )

    def forward(self, x, bboxes):
        num_level_assignments = len(self.level_poolers)
        assert len(x) == num_level_assignments and len(bboxes) == x[0].size(0)

        pooler_fmt_boxes = convert_boxes_to_pooler_format(bboxes)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            bboxes, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        batches = pooler_fmt_boxes.shape[0]
        channels = x[0].shape[1]
        output_size = self.output_size[0]
        sizes = (batches, channels, output_size, output_size)

        output = torch.zeros(sizes, dtype=x[0].dtype, device=x[0].device)

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = (level_assignments == level).nonzero(as_tuple=True)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x_level, pooler_fmt_boxes_level))

        return output
