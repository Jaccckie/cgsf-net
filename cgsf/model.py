import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import trunc_normal_
from .sparsevit import SparseViT
from .backbone import build_dinov3
from .fusion import DINOv3TensorNormalizer, CLSGuidedGlobalAggregator, SmallOffsetFullCrossAttentionCLSGateThirdFusion
from .decoder import FusionConfidenceEstimator, ConfidenceGuidedSpatialDynamicScaleBoundaryDecoder

class ThirdOnlyOffsetFullCrossAttentionCLSGateFusionModel(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        dino_config_path=None,
        dino_bridge_dim: int = 512,
        freeze_dinov3: bool = True,
        align_dim: int = 160,
        attn_dim: int = 320,
        num_heads: int = 8,
        cls_gate_hidden_dim: int = 512,
        global_hidden_dim: int = 1024,
        global_num_heads: int = 8,
        max_offset: float = 0.15,
        decoder_dim: int = 256,
        decoder_condition_hidden_dim: int = 512,
        confidence_hidden_dim: int = 128,
        spatial_hidden_dim: int = 192,
        decoder_scale_temperature: float = 1.0,
        spatial_logit_scale: float = 1.0,
        boundary_residual_init: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.freeze_dinov3 = freeze_dinov3
        self.dino_bridge_dim = dino_bridge_dim

        self.sparsevit = SparseViT(
            layers=[5, 8, 20, 7],
            embed_dim=[64, 128, 320, 512],
            img_size=img_size,
            s_blocks3=[8, 4, 2, 1],
            s_blocks4=[2, 1],
            head_dim=64,
            drop_path_rate=0.2,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pretrained_path=None,
        )

        self.dinov3_model = build_dinov3(dino_config_path)

        if freeze_dinov3:
            for p in self.dinov3_model.parameters():
                p.requires_grad = False
            self.dinov3_model.eval()

        dino_config = getattr(self.dinov3_model, "config", None)
        self.dino_patch_size = getattr(dino_config, "patch_size", 16)
        self.num_registers = getattr(dino_config, "num_register_tokens", 4)
        self.dino_hidden_size = getattr(dino_config, "hidden_size", 4096)

        self.dino_normalizer = DINOv3TensorNormalizer(
            None,
            patch_size=self.dino_patch_size,
        )

        self.dino_patch_proj = nn.Conv2d(
            self.dino_hidden_size,
            dino_bridge_dim,
            kernel_size=1,
            bias=False
        )

        self.global_aggregator = CLSGuidedGlobalAggregator(
            token_dim=self.dino_hidden_size,
            out_dim=dino_bridge_dim,
            hidden_dim=global_hidden_dim,
            num_heads=global_num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        self.third_fusion = SmallOffsetFullCrossAttentionCLSGateThirdFusion(
            sparse_dim=320,
            dino_dim=dino_bridge_dim,
            cls_dim=self.dino_hidden_size,
            align_dim=align_dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            global_dim=dino_bridge_dim,
            cls_gate_hidden_dim=cls_gate_hidden_dim,
            max_offset=max_offset,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        self.confidence_estimator = FusionConfidenceEstimator(
            feature_dim=320,
            cls_dim=self.dino_hidden_size,
            global_dim=dino_bridge_dim,
            hidden_dim=confidence_hidden_dim,
        )
        self.decoder = ConfidenceGuidedSpatialDynamicScaleBoundaryDecoder(
            in_channels=(320, 320, 320, 320, 512, 512),
            decoder_dim=decoder_dim,
            cls_dim=self.dino_hidden_size,
            global_dim=dino_bridge_dim,
            condition_hidden_dim=decoder_condition_hidden_dim,
            confidence_hidden_dim=confidence_hidden_dim,
            spatial_hidden_dim=spatial_hidden_dim,
            predict_channels=1,
            scale_temperature=decoder_scale_temperature,
            spatial_logit_scale=spatial_logit_scale,
            boundary_residual_init=boundary_residual_init,
        )
        self._init_new_modules()
        self.confidence_estimator.apply(self._init_weights)
        self.confidence_estimator._init_confidence_gate()
        self.decoder._init_gates()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def _init_new_modules(self):
        modules = [
            self.dino_patch_proj,
            self.global_aggregator,
            self.third_fusion,
            self.decoder,
        ]
        for module in modules:
            module.apply(self._init_weights)

        nn.init.zeros_(self.third_fusion.offset_net[-1].weight)
        nn.init.zeros_(self.third_fusion.offset_net[-1].bias)
        nn.init.constant_(self.third_fusion.alpha, 0.01)

    def extract_dinov3_features(self, pixel_values: torch.Tensor):
        if self.freeze_dinov3:
            with torch.no_grad():
                outputs = self.dinov3_model(pixel_values=pixel_values)
        else:
            outputs = self.dinov3_model(pixel_values=pixel_values)

        tokens = outputs.last_hidden_state
        B, _, H, W = pixel_values.shape

        grid_h = H // self.dino_patch_size
        grid_w = W // self.dino_patch_size
        num_patches = grid_h * grid_w

        expected_tokens = 1 + self.num_registers + num_patches
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"DINOv3 token 数不匹配：实际为 {tokens.shape[1]}，"
                f"期望为 1 + {self.num_registers} + {num_patches} = {expected_tokens}"
            )

        cls_token = tokens[:, 0, :]
        reg_tokens = tokens[:, 1:1 + self.num_registers, :]
        patch_tokens = tokens[:, 1 + self.num_registers:, :]

        patch_2d = patch_tokens.transpose(1, 2).reshape(
            B, self.dino_hidden_size, grid_h, grid_w
        )

        return cls_token, reg_tokens, patch_2d

    def build_global_context(self, cls_token: torch.Tensor, reg_tokens: torch.Tensor, return_debug: bool = False):
        global_ctx, global_debug = self.global_aggregator(
            cls_token=cls_token,
            reg_tokens=reg_tokens,
            return_debug=return_debug,
        )
        return global_ctx, global_debug

    def forward(
        self,
        image: torch.Tensor,
        image_raw: torch.Tensor = None,
        return_debug: bool = False,
        return_logits: bool = False,
    ):
        sparsevit_features = self.sparsevit(image)

        required_keys = ['third1', 'third2', 'third3', 'third', 'last1', 'last']
        for key in required_keys:
            if key not in sparsevit_features:
                raise KeyError(f"SparseViT 输出缺少必要键：{key}")

        dino_source = image if image_raw is None else image_raw
        dino_input = self.dino_normalizer(dino_source)

        cls_token, reg_tokens, dino_patch_2d_raw = self.extract_dinov3_features(dino_input)
        dino_patch_2d = self.dino_patch_proj(dino_patch_2d_raw)
        global_ctx, global_token_debug = self.build_global_context(
            cls_token, reg_tokens, return_debug=return_debug
        )

        third_feat = sparsevit_features['third']

        if dino_patch_2d.shape[-2:] != third_feat.shape[-2:]:
            raise ValueError(
                f"只融合 third 的方案要求 DINO patch 网格与 third 尺寸天然一致，"
                f"当前 DINO={dino_patch_2d.shape[-2:]}, third={third_feat.shape[-2:]}"
            )

        third_feat_fused, fusion_debug = self.third_fusion(
            sparse_feat=third_feat,
            dino_feat=dino_patch_2d,
            global_ctx=global_ctx,
            cls_token=cls_token,
            return_debug=return_debug,
        )

        confidence_map, confidence_feature, confidence_debug = self.confidence_estimator(
            third_feat=third_feat,
            third_feat_fused=third_feat_fused,
            cls_token=cls_token,
            global_ctx=global_ctx,
            return_debug=return_debug,
        )

        feature_list = [
            sparsevit_features['third1'],
            sparsevit_features['third2'],
            sparsevit_features['third3'],
            third_feat_fused,
            sparsevit_features['last1'],
            sparsevit_features['last'],
        ]

        logits, boundary_logits, decoder_debug = self.decoder(
            feature_list,
            cls_token=cls_token,
            global_ctx=global_ctx,
            confidence_map=confidence_map,
            confidence_feature=confidence_feature,
            return_debug=return_debug,
        )
        logits = F.interpolate(
            logits,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        boundary_logits = F.interpolate(
            boundary_logits,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        pred = torch.sigmoid(logits)

        if return_debug:
            debug = {}
            for values in (fusion_debug, global_token_debug, confidence_debug, decoder_debug):
                if values is not None:
                    debug.update(values)
            return (logits if return_logits else pred), debug
        return logits if return_logits else pred


def create_third_only_offset_full_cross_attention_cls_gate_fusion_model(**kwargs):
    return ThirdOnlyOffsetFullCrossAttentionCLSGateFusionModel(**kwargs)
