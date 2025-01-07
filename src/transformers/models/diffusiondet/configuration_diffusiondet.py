from transformers import PretrainedConfig

from transformers.models.auto import CONFIG_MAPPING
from transformers.utils.backbone_utils import verify_backbone_config_arguments

from ...utils import logging, PushToHubMixin

logger = logging.get_logger(__name__)

CONFIG_NAME = "config.json"

class DiffusionDetConfig(PretrainedConfig):

    config_name = CONFIG_NAME

    def __init__(
            self,
            use_timm_backbone=True,
            backbone_config=None,
            num_channels=3,
            pixel_mean=(123.675, 116.280, 103.530),
            pixel_std=(58.395, 57.120, 57.375),
            resnet_out_features=("res2", "res3", "res4", "res5"),
            resnet_in_features=("res2", "res3", "res4", "res5"),
            roi_head_in_features=("p2", "p3", "p4", "p5"),
            fpn_out_channels=256,
            pooler_resolution=7,
            sampling_ratio=2,
            num_proposals=300,
            num_attn_heads=8,
            dropout=0.0,
            dim_feedforward=2048,
            activation="relu",
            hidden_dim=256,
            num_cls=1,
            num_reg=3,
            num_heads=6,
            num_dynamic=2,
            dim_dynamic=64,
            class_weight=2.0,
            giou_weight=2.0,
            l1_weight=5.0,
            deep_supervision=True,
            no_object_weight=0.1,
            use_focal=True,
            use_fed_loss=False,
            alpha=0.25,
            gamma=2.0,
            prior_prob=0.01,
            ota_k=5,
            snr_scale=2.0,
            sample_step=1,
            use_nms=True,
            swin_size="B",
            use_swin_checkpoint=False,
            swin_out_features=(0, 1, 2, 3),
            optimizer="ADAMW",
            backbone_multiplier=1.0,
            backbone='resnet50',
            use_pretrained_backbone=True,
            backbone_kwargs=None,
            dilation=False,
            **kwargs
    ):
        # We default to values which were previously hard-coded in the model. This enables configurability of the config
        # while keeping the default behavior the same.
        if use_timm_backbone and backbone_kwargs is None:
            backbone_kwargs = {}
            if dilation:
                backbone_kwargs["output_stride"] = 16
            backbone_kwargs["out_indices"] = [1, 2, 3, 4]
            backbone_kwargs["in_chans"] = num_channels
        # Backwards compatibility
        elif not use_timm_backbone and backbone in (None, "resnet50"):
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
            backbone = None
            # set timm attributes to None
            dilation = None

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        # Backbone.
        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_channels = num_channels
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.backbone_kwargs = backbone_kwargs
        self.dilation = dilation
        self.fpn_out_channels = fpn_out_channels

        # Model.
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.resnet_out_features = resnet_out_features
        self.resnet_in_features = resnet_in_features
        self.roi_head_in_features = roi_head_in_features
        self.pooler_resolution = pooler_resolution
        self.sampling_ratio = sampling_ratio
        self.num_proposals = num_proposals

        # RCNN Head.
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.num_cls = num_cls
        self.num_reg = num_reg
        self.num_heads = num_heads

        # Dynamic Conv.
        self.num_dynamic = num_dynamic
        self.dim_dynamic = dim_dynamic

        # Loss.
        self.class_weight = class_weight
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight
        self.deep_supervision = deep_supervision
        self.no_object_weight = no_object_weight

        # Focal Loss.
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.alpha = alpha
        self.gamma = gamma
        self.prior_prob = prior_prob

        # Dynamic K
        self.ota_k = ota_k

        # Diffusion
        self.snr_scale = snr_scale
        self.sample_step = sample_step

        # Inference
        self.use_nms = use_nms

        # Swin Backbones
        self.swin_size = swin_size
        self.use_swin_checkpoint = use_swin_checkpoint
        self.swin_out_features = swin_out_features

        # Optimizer.
        self.optimizer = optimizer
        self.backbone_multiplier = backbone_multiplier

        self.num_labels = 80

        super().__init__()
