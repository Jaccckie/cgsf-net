"""Build DINOv3 from a bundled config; weights come only from the fusion checkpoint."""
from .paths import DEFAULT_DINO_CONFIG, resolve_relative

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding


def build_dinov3(config_path=None):
    config_path = resolve_relative(config_path if config_path is not None else DEFAULT_DINO_CONFIG)
    config = DINOv3ViTConfig.from_json_file(str(config_path))
    # Avoid allocating and randomly initializing a second 7B backbone.
    with torch.device("meta"):
        model = DINOv3ViTModel(config)
    # RoPE frequencies are non-persistent: they are absent from state_dict.
    # Reconstruct them on CPU, using the installed Transformers implementation.
    for module in model.modules():
        if isinstance(module, DINOv3ViTRopePositionEmbedding):
            reference = DINOv3ViTRopePositionEmbedding(config)
            for name in module._non_persistent_buffers_set:
                module.register_buffer(name, getattr(reference, name), persistent=False)
    return model
