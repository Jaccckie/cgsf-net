import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_group_num(channels: int, prefer: int = 8) -> int:
    for g in range(min(prefer, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class DINOv3TensorNormalizer(nn.Module):
    def __init__(self, processor, patch_size: int = 16):
        super().__init__()

        mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False
        )
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"DINOv3 输入必须是 [B, 3, H, W]，当前为 {tuple(x.shape)}")

        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"DINOv3 输入高宽必须是 patch_size={self.patch_size} 的整数倍，"
                f"当前 H={x.shape[-2]}, W={x.shape[-1]}"
            )

        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        return x


class CLSDeltaGate(nn.Module):
    def __init__(self, cls_dim: int, target_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cls_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.net(cls_token))
        return gate.unsqueeze(-1).unsqueeze(-1)


class CLSGuidedGlobalAggregator(nn.Module):
    def __init__(
        self,
        token_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim={token_dim} 必须能被 num_heads={num_heads} 整除")

        self.token_dim = token_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_tokens = nn.LayerNorm(token_dim)
        self.norm_cls = nn.LayerNorm(token_dim)

        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, token_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)

        self.out_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def _reshape_heads(self, x: torch.Tensor, B: int, N: int):
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, cls_token: torch.Tensor, reg_tokens: torch.Tensor, return_debug: bool = False):
        B, R, C = reg_tokens.shape

        all_tokens = torch.cat([cls_token.unsqueeze(1), reg_tokens], dim=1)
        all_tokens = self.norm_tokens(all_tokens)

        cls_q = self.norm_cls(cls_token).unsqueeze(1)

        q = self.q_proj(cls_q)
        k = self.k_proj(all_tokens)
        v = self.v_proj(all_tokens)

        q = self._reshape_heads(q, B, 1)
        k = self._reshape_heads(k, B, 1 + R)
        v = self._reshape_heads(v, B, 1 + R)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        pooled = torch.matmul(attn, v)
        pooled = pooled.permute(0, 2, 1, 3).contiguous().view(B, C)

        global_ctx = self.out_proj(pooled)

        if return_debug:
            debug_dict = {
                "global_attn_mean": attn.mean().detach(),
                "global_attn_max": attn.max().detach(),
                "global_attn_cls_weight_mean": attn[..., 0].mean().detach(),
            }

            if R > 0:
                debug_dict["global_attn_reg_weight_mean"] = attn[..., 1:].mean().detach()
            else:
                debug_dict["global_attn_reg_weight_mean"] = torch.zeros(
                    (), device=attn.device, dtype=attn.dtype
                )

            return global_ctx, debug_dict

        return global_ctx, None


class SmallOffsetFullCrossAttentionCLSGateThirdFusion(nn.Module):
    def __init__(
        self,
        sparse_dim: int = 320,
        dino_dim: int = 512,
        cls_dim: int = 4096,
        align_dim: int = 160,
        attn_dim: int = 320,
        num_heads: int = 8,
        global_dim: int = 512,
        cls_gate_hidden_dim: int = 512,
        max_offset: float = 0.15,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError(f"attn_dim={attn_dim} 必须能被 num_heads={num_heads} 整除")

        self.sparse_dim = sparse_dim
        self.dino_dim = dino_dim
        self.cls_dim = cls_dim
        self.align_dim = align_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.global_dim = global_dim
        self.max_offset = max_offset

        gn = _safe_group_num(sparse_dim, 8)

        self.sparse_align_proj = nn.Conv2d(sparse_dim, align_dim, kernel_size=1, bias=False)
        self.dino_align_proj = nn.Conv2d(dino_dim, align_dim, kernel_size=1, bias=False)

        self.offset_net = nn.Sequential(
            nn.Conv2d(align_dim * 2, align_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(align_dim, 2, kernel_size=3, padding=1, bias=True),
        )

        self.q_proj = nn.Conv2d(sparse_dim, attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dino_dim, attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dino_dim, attn_dim, kernel_size=1, bias=False)

        self.q_norm = nn.LayerNorm(attn_dim)
        self.k_norm = nn.LayerNorm(attn_dim)
        self.v_norm = nn.LayerNorm(attn_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_proj = nn.Conv2d(attn_dim, sparse_dim, kernel_size=1, bias=True)

        self.local_refine = nn.Sequential(
            nn.Conv2d(sparse_dim, sparse_dim, kernel_size=3, padding=1, groups=sparse_dim, bias=False),
            nn.GELU(),
            nn.Conv2d(sparse_dim, sparse_dim, kernel_size=1, bias=True),
            nn.GroupNorm(gn, sparse_dim),
            nn.GELU(),
        )

        self.context_gate = nn.Sequential(
            nn.Linear(global_dim, sparse_dim),
            nn.GELU(),
            nn.Linear(sparse_dim, sparse_dim),
        )

        self.cls_delta_gate = CLSDeltaGate(
            cls_dim=cls_dim,
            target_dim=sparse_dim,
            hidden_dim=cls_gate_hidden_dim,
        )

        self.alpha = nn.Parameter(torch.ones(1) * 0.01)

        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    @staticmethod
    def _build_base_grid(B: int, H: int, W: int, device, dtype):
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H * 2 - 1
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W * 2 - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).expand(B, H, W, 2).contiguous()

    def _reshape_heads(self, x: torch.Tensor, B: int, N: int):
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        sparse_feat: torch.Tensor,
        dino_feat: torch.Tensor,
        global_ctx: torch.Tensor,
        cls_token: torch.Tensor,
        return_debug: bool = False,
    ):
        B, _, H, W = sparse_feat.shape
        Bd, _, Hd, Wd = dino_feat.shape

        if B != Bd:
            raise ValueError(f"batch 不一致: sparse={B}, dino={Bd}")

        if (Hd, Wd) != (H, W):
            raise ValueError(
                f"当前实现要求 DINO third 与 SparseViT third 空间尺寸一致，"
                f"当前 DINO={(Hd, Wd)}, SparseViT={(H, W)}"
            )

        if not getattr(self, "use_cross_attention", True):
            return sparse_feat, None

        if getattr(self, "use_offset_alignment", True):
            sparse_align = self.sparse_align_proj(sparse_feat)
            dino_align = self.dino_align_proj(dino_feat)
            offset = self.offset_net(torch.cat([sparse_align, dino_align], dim=1))
            offset = torch.tanh(offset) * self.max_offset
            base_grid = self._build_base_grid(B, H, W, sparse_feat.device, sparse_feat.dtype)
            sample_grid = base_grid + offset.permute(0, 2, 3, 1)
            dino_aligned = F.grid_sample(
                dino_feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=False,
            )
        else:
            offset = sparse_feat.new_zeros(B, 2, H, W)
            dino_aligned = dino_feat

        q_map = self.q_proj(sparse_feat)
        k_map = self.k_proj(dino_aligned)
        v_map = self.v_proj(dino_aligned)

        Nq = H * W
        Nk = H * W

        q = q_map.flatten(2).transpose(1, 2)
        k = k_map.flatten(2).transpose(1, 2)
        v = v_map.flatten(2).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q = self._reshape_heads(q, B, Nq)
        k = self._reshape_heads(k, B, Nk)
        v = self._reshape_heads(v, B, Nk)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        z = torch.matmul(attn, v)
        z = z.permute(0, 2, 1, 3).contiguous().view(B, Nq, self.attn_dim)
        z = z.transpose(1, 2).reshape(B, self.attn_dim, H, W)

        delta = self.out_proj(z)
        delta = self.proj_drop(delta)
        delta = self.local_refine(delta)

        global_gate = torch.sigmoid(self.context_gate(global_ctx)).view(B, self.sparse_dim, 1, 1)
        delta = delta * global_gate

        cls_gate = self.cls_delta_gate(cls_token)
        delta = delta * cls_gate

        fusion_scale = 0.1 * torch.tanh(self.alpha)
        output = sparse_feat + fusion_scale * delta

        if return_debug:
            return output, {
                "fusion_scale": fusion_scale.detach(),
                "offset_abs_mean": offset.abs().mean().detach(),
                "offset_abs_max": offset.abs().max().detach(),
                "attn_mean": attn.mean().detach(),
                "attn_max_mean": attn.max(dim=-1).values.mean().detach(),
                "cls_gate_mean": cls_gate.mean().detach(),
                "cls_gate_std": cls_gate.std().detach(),
                "global_gate_mean": global_gate.mean().detach(),
                "global_gate_std": global_gate.std().detach(),
            }

        return output, None
