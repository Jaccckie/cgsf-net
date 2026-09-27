import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import _safe_group_num

class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_safe_group_num(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CLSConditionProjector(nn.Module):
    def __init__(
        self,
        cls_dim: int,
        global_dim: int,
        hidden_dim: int,
        drop: float = 0.0,
    ):
        super().__init__()
        half_hidden = hidden_dim // 2
        self.cls_proj = nn.Sequential(
            nn.LayerNorm(cls_dim),
            nn.Linear(cls_dim, half_hidden),
            nn.GELU(),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, half_hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(half_hidden * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, cls_token: torch.Tensor, global_ctx: torch.Tensor) -> torch.Tensor:
        cls_cond = self.cls_proj(cls_token)
        global_cond = self.global_proj(global_ctx)
        return self.fuse(torch.cat([cls_cond, global_cond], dim=1))


class FusionConfidenceEstimator(nn.Module):
    def __init__(
        self,
        feature_dim: int = 320,
        cls_dim: int = 4096,
        global_dim: int = 512,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.orig_proj = ConvGNAct(feature_dim, hidden_dim, kernel_size=1)
        self.fused_proj = ConvGNAct(feature_dim, hidden_dim, kernel_size=1)
        self.delta_proj = ConvGNAct(feature_dim, hidden_dim, kernel_size=1)

        self.condition_projector = CLSConditionProjector(
            cls_dim=cls_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim * 2,
        )
        self.condition_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mix = nn.Sequential(
            ConvGNAct(hidden_dim * 3, hidden_dim, kernel_size=3),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3),
        )
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self._init_confidence_gate()

    def _init_confidence_gate(self):
        for module in self.condition_gate:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        third_feat: torch.Tensor,
        third_feat_fused: torch.Tensor,
        cls_token: torch.Tensor,
        global_ctx: torch.Tensor,
        return_debug: bool = False,
    ):
        delta_feat = torch.abs(third_feat_fused - third_feat)

        orig = self.orig_proj(third_feat)
        fused = self.fused_proj(third_feat_fused)
        delta = self.delta_proj(delta_feat)

        condition = self.condition_projector(cls_token=cls_token, global_ctx=global_ctx)
        condition_gate = torch.sigmoid(self.condition_gate(condition)).view(
            condition.shape[0], self.hidden_dim, 1, 1
        )

        confidence_feature = self.mix(
            torch.cat(
                [
                    orig,
                    fused * (0.5 + condition_gate),
                    delta * (1.0 + condition_gate),
                ],
                dim=1,
            )
        )
        confidence_logits = self.head(confidence_feature)
        confidence_map = torch.sigmoid(confidence_logits)

        if return_debug:
            return confidence_map, confidence_feature, {
                "confidence_map_mean": confidence_map.mean().detach(),
                "confidence_map_std": confidence_map.std().detach(),
                "confidence_logit_abs_mean": confidence_logits.abs().mean().detach(),
                "confidence_condition_gate_mean": condition_gate.mean().detach(),
            }

        return confidence_map, confidence_feature, None


class ConfidenceGuidedSpatialDynamicScaleBoundaryDecoder(nn.Module):
    def __init__(
        self,
        in_channels=(320, 320, 320, 320, 512, 512),
        decoder_dim: int = 256,
        cls_dim: int = 4096,
        global_dim: int = 512,
        condition_hidden_dim: int = 512,
        confidence_hidden_dim: int = 128,
        spatial_hidden_dim: int = 192,
        predict_channels: int = 1,
        scale_temperature: float = 1.0,
        spatial_logit_scale: float = 1.0,
        boundary_residual_init: float = 0.1,
    ):
        super().__init__()

        if scale_temperature <= 0:
            raise ValueError(f"scale_temperature 必须大于 0，当前为 {scale_temperature}")

        self.in_channels = tuple(in_channels)
        self.num_scales = len(self.in_channels)
        self.decoder_dim = decoder_dim
        self.scale_temperature = scale_temperature
        self.spatial_logit_scale = spatial_logit_scale

        self.proj_layers = nn.ModuleList([
            ConvGNAct(in_channel, decoder_dim, kernel_size=1)
            for in_channel in self.in_channels
        ])

        self.condition_projector = CLSConditionProjector(
            cls_dim=cls_dim,
            global_dim=global_dim,
            hidden_dim=condition_hidden_dim,
        )

        self.image_scale_gate = nn.Linear(condition_hidden_dim, self.num_scales)
        self.global_channel_gate = nn.Linear(condition_hidden_dim, decoder_dim)

        self.local_context_net = nn.Sequential(
            ConvGNAct(decoder_dim * 3, decoder_dim, kernel_size=3),
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.semantic_context_net = nn.Sequential(
            ConvGNAct(decoder_dim * 3, decoder_dim, kernel_size=3),
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3),
        )

        self.confidence_reduce = nn.Sequential(
            ConvGNAct(confidence_hidden_dim + 1, decoder_dim, kernel_size=1),
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.confidence_channel_gate = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_context_net = nn.Sequential(
            ConvGNAct(decoder_dim * 4, spatial_hidden_dim, kernel_size=3),
            ConvGNAct(spatial_hidden_dim, spatial_hidden_dim, kernel_size=3),
        )
        self.spatial_scale_head = nn.Conv2d(spatial_hidden_dim, self.num_scales, kernel_size=1)

        self.context_refine = nn.Sequential(
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3),
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.coarse_head = nn.Conv2d(decoder_dim, predict_channels, kernel_size=1)

        self.boundary_refine = nn.Sequential(
            ConvGNAct(decoder_dim * 5, decoder_dim, kernel_size=3),
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3),
        )
        self.boundary_head = nn.Conv2d(decoder_dim, predict_channels, kernel_size=1)
        self.boundary_delta_head = nn.Conv2d(decoder_dim, predict_channels, kernel_size=1)
        self.boundary_residual_scale = nn.Parameter(torch.tensor(float(boundary_residual_init)))

        self._init_gates()

    def _init_gates(self):
        nn.init.zeros_(self.image_scale_gate.weight)
        nn.init.zeros_(self.image_scale_gate.bias)
        nn.init.zeros_(self.global_channel_gate.weight)
        nn.init.zeros_(self.global_channel_gate.bias)
        nn.init.zeros_(self.spatial_scale_head.weight)
        if self.spatial_scale_head.bias is not None:
            nn.init.zeros_(self.spatial_scale_head.bias)

        for module in self.confidence_channel_gate:
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features,
        cls_token: torch.Tensor,
        global_ctx: torch.Tensor,
        confidence_map: torch.Tensor,
        confidence_feature: torch.Tensor,
        return_debug: bool = False,
    ):
        if len(features) != self.num_scales:
            raise ValueError(f"期望 {self.num_scales} 个尺度特征，实际收到 {len(features)} 个")

        target_size = features[0].shape[-2:]
        projected = []
        for feature, proj in zip(features, self.proj_layers):
            feature = proj(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            projected.append(feature)

        if confidence_map.shape[-2:] != target_size:
            confidence_map = F.interpolate(confidence_map, size=target_size, mode='bilinear', align_corners=False)
        if confidence_feature.shape[-2:] != target_size:
            confidence_feature = F.interpolate(
                confidence_feature,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )

        condition = self.condition_projector(cls_token=cls_token, global_ctx=global_ctx)

        image_scale_logits = self.image_scale_gate(condition) / self.scale_temperature
        global_channel_gate = 2.0 * torch.sigmoid(self.global_channel_gate(condition)).view(
            -1, self.decoder_dim, 1, 1
        )

        confidence_context = self.confidence_reduce(
            torch.cat([confidence_feature, confidence_map], dim=1)
        )
        confidence_channel_gate = 0.5 + self.confidence_channel_gate(confidence_context)

        local_branch = self.local_context_net(torch.cat(projected[:3], dim=1))
        semantic_branch = self.semantic_context_net(torch.cat(projected[3:], dim=1))
        confidence_balance = confidence_map
        branch_fused = (1.0 - confidence_balance) * local_branch + confidence_balance * semantic_branch

        spatial_context = self.spatial_context_net(
            torch.cat(
                [
                    projected[0],
                    projected[3],
                    projected[5],
                    confidence_context,
                ],
                dim=1,
            )
        )
        spatial_scale_logits = self.spatial_scale_head(spatial_context)

        combined_scale_logits = (
            image_scale_logits.view(-1, self.num_scales, 1, 1)
            + self.spatial_logit_scale * spatial_scale_logits
        )
        spatial_scale_weights = torch.softmax(combined_scale_logits, dim=1)

        stacked = torch.stack(projected, dim=1)
        dynamic_fused = (stacked * spatial_scale_weights.unsqueeze(2)).sum(dim=1)

        uncertainty_map = 1.0 - torch.abs(2.0 * confidence_balance - 1.0)
        dynamic_fused = dynamic_fused + branch_fused + uncertainty_map * confidence_context
        dynamic_fused = dynamic_fused * global_channel_gate * confidence_channel_gate
        dynamic_fused = self.context_refine(dynamic_fused)

        coarse_logits = self.coarse_head(dynamic_fused)

        boundary_feature = self.boundary_refine(
            torch.cat(
                [
                    projected[0],
                    projected[1],
                    projected[3],
                    dynamic_fused,
                    confidence_context,
                ],
                dim=1,
            )
        )
        boundary_logits = self.boundary_head(boundary_feature)
        boundary_gate = torch.sigmoid(boundary_logits)
        boundary_delta = self.boundary_delta_head(boundary_feature)
        residual_scale = torch.tanh(self.boundary_residual_scale)
        refined_logits = coarse_logits + residual_scale * boundary_gate * (1.0 + uncertainty_map) * boundary_delta

        if return_debug:
            debug_dict = {
                "decoder_global_channel_gate_mean": global_channel_gate.mean().detach(),
                "decoder_global_channel_gate_std": global_channel_gate.std().detach(),
                "decoder_confidence_channel_gate_mean": confidence_channel_gate.mean().detach(),
                "decoder_confidence_channel_gate_std": confidence_channel_gate.std().detach(),
                "decoder_confidence_balance_mean": confidence_balance.mean().detach(),
                "decoder_uncertainty_mean": uncertainty_map.mean().detach(),
                "decoder_spatial_entropy": (
                    -(spatial_scale_weights.clamp_min(1e-8) * spatial_scale_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
                ).detach(),
                "decoder_boundary_gate_mean": boundary_gate.mean().detach(),
                "decoder_boundary_delta_abs_mean": boundary_delta.abs().mean().detach(),
                "decoder_boundary_residual_scale": residual_scale.detach(),
            }
            image_scale_weights = torch.softmax(image_scale_logits, dim=1)
            for idx in range(self.num_scales):
                debug_dict[f"decoder_image_scale_weight_{idx}"] = image_scale_weights[:, idx].mean().detach()
                debug_dict[f"decoder_spatial_scale_weight_{idx}"] = spatial_scale_weights[:, idx].mean().detach()

            return refined_logits, boundary_logits, debug_dict

        return refined_logits, boundary_logits, None
