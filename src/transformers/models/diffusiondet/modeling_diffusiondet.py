import math
import random
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from transformers import PreTrainedModel
import wandb

from transformers.utils.backbone_utils import load_backbone
from .configuration_diffusiondet import DiffusionDetConfig

from .head import HeadDynamicK
from .loss import CriterionDynamicK

from ...utils import ModelOutput

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

@dataclass
class DiffusionDetOutput(ModelOutput):
    """
    Output type of DiffusionDet.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    labels: torch.IntTensor = None
    pred_boxes: torch.FloatTensor = None

class DiffusionDet(PreTrainedModel):
    """
    Implement DiffusionDet
    """
    config_class = DiffusionDetConfig
    main_input_name = "pixel_values"

    def __init__(self, config):
        super(DiffusionDet, self).__init__(config)

        self.in_features = config.roi_head_in_features
        self.num_classes = config.num_labels
        self.num_proposals = config.num_proposals
        self.num_heads = config.num_heads

        self.backbone = load_backbone(config)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.backbone.channels,
            out_channels=config.fpn_out_channels,
            # extra_blocks=LastLevelMaxPool(),
        )

        # build diffusion
        betas = cosine_beta_schedule(1000)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)

        timesteps, = betas.shape
        sampling_timesteps = config.sample_step

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.ddim_sampling_eta = 1.
        self.scale = config.snr_scale
        assert self.sampling_timesteps <= timesteps

        roi_input_shape = {
            'p2': {'stride': 4},
            'p3': {'stride': 8},
            'p4': {'stride': 16},
            'p5': {'stride': 32},
            'p6': {'stride': 64}
        }
        self.head = HeadDynamicK(config, roi_input_shape=roi_input_shape)

        self.deep_supervision = config.deep_supervision
        self.use_focal = config.use_focal
        self.use_fed_loss = config.use_fed_loss
        self.use_nms = config.use_nms

        weight_dict = {
            "loss_ce": config.class_weight, "loss_bbox": config.l1_weight, "loss_giou": config.giou_weight
        }
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = CriterionDynamicK(config, num_classes=self.num_classes, weight_dict=weight_dict)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = ops.box_convert(x_boxes, 'cxcywh', 'xyxy')
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = ops.box_convert(x_start, 'xyxy', 'cxcywh')
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh):
        bs = len(batched_inputs)
        image_sizes = batched_inputs.shape
        shape = (bs, self.num_proposals, 4)

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        outputs_class, outputs_coord = None, None
        for time, time_next in time_pairs:
            time_cond = torch.full((bs,), time, device=self.device, dtype=torch.long)

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
            threshold = 0.5
            score_per_image = torch.sigmoid(score_per_image)
            value, _ = torch.max(score_per_image, -1, keepdim=False)
            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            pred_noise = pred_noise[:, keep_idx, :]
            x_start = x_start[:, keep_idx, :]
            img = img[:, keep_idx, :]

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)

            if self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1])
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)

            if self.use_nms:
                keep = ops.batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            return box_pred_per_image, scores_per_image, labels_per_image
        else:
            return self.inference(outputs_class[-1], outputs_coord[-1])

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, pixel_values, labels):
        """
        Args:
        """
        images = pixel_values.to(self.device)
        images_whwh = list()
        for image in images:
            h, w = image.shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], device=self.device))
        images_whwh = torch.stack(images_whwh)

        features = self.backbone(images)
        features = OrderedDict(
            [(key, feature) for key, feature in zip(self.backbone.out_features, features.feature_maps)]
        )
        features = self.fpn(features)  # [144, 72, 36, 18]
        features = [features[f] for f in features.keys()]

        # if self.training:
        labels = list(map(lambda tensor: tensor.to(self.device), labels))
        targets, x_boxes, noises, ts = self.prepare_targets(labels)

        ts = ts.squeeze(-1)
        x_boxes = x_boxes * images_whwh[:, None, :]

        outputs_class, outputs_coord = self.head(features, x_boxes, ts)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.deep_supervision:
            output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                     for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        loss_dict['loss'] = sum([loss_dict[k] for k in weight_dict.keys()])

        wandb_logs_values = ["loss_ce", "loss_bbox", "loss_giou"]

        if self.training:
            wandb.log({f'train/{k}': v.detach().cpu().numpy() for k, v in loss_dict.items() if k in wandb_logs_values})
        else:
            wandb.log({f'eval/{k}': v.detach().cpu().numpy() for k, v in loss_dict.items() if k in wandb_logs_values})

        if not self.training:
            pred_logits, pred_labels, pred_boxes  = self.ddim_sample(pixel_values, features, images_whwh)
            return DiffusionDetOutput(
                loss=loss_dict['loss'],
                loss_dict=loss_dict,
                logits=pred_logits,
                labels=pred_labels,
                pred_boxes=pred_boxes,
            )

        return DiffusionDetOutput(
            loss=loss_dict['loss'],
            loss_dict=loss_dict,
            logits=output['pred_logits'],
            pred_boxes=output['pred_boxes']
        )

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = ops.box_convert(x, 'cxcywh', 'xyxy')

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for target in targets:
            h, w = target.size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = target.class_labels.to(self.device)
            gt_boxes = target.boxes.to(self.device)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            gt_boxes = gt_boxes * image_size_xyxy
            gt_boxes = ops.box_convert(gt_boxes, 'cxcywh', 'xyxy')

            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            new_targets.append({
                "labels": gt_classes,
                "boxes": target.boxes.to(self.device),
                "boxes_xyxy": gt_boxes,
                "image_size_xyxy": image_size_xyxy.to(self.device),
                "image_size_xyxy_tgt": image_size_xyxy_tgt.to(self.device),
                "area": ops.box_area(target.boxes.to(self.device)),
            })

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def inference(self, box_cls, box_pred):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        boxes_output = []
        logits_output = []
        labels_output = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image) in enumerate(zip(
                    scores, box_pred
            )):
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = ops.batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                boxes_output.append(box_pred_per_image)
                logits_output.append(scores_per_image)
                labels_output.append(labels_per_image)
        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image) in enumerate(zip(
                    scores, labels, box_pred
            )):
                if self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = ops.batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                boxes_output.append(box_pred_per_image)
                logits_output.append(scores_per_image)
                labels_output.append(labels_per_image)

        return boxes_output, logits_output, labels_output
