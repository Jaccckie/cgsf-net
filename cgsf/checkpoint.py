"""Strict, memory-mapped loading of the complete epoch-40 inference network."""
import argparse
import inspect

import torch

from .model import ThirdOnlyOffsetFullCrossAttentionCLSGateFusionModel

from .paths import DEFAULT_CHECKPOINT, resolve_relative


def read_checkpoint(path):
    path = resolve_relative(path)
    # The original file contains argparse.Namespace in addition to tensors.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return checkpoint


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device='cpu', **overrides):
    """Load all 1640 model entries strictly, and return an eval-only model.

    Constructor options are recovered from checkpoint.args; overrides are explicit.
    No optimizer/scaler state is restored and no external pretraining file is read.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    state = checkpoint.get('model', checkpoint)
    saved = checkpoint.get('args', {})
    saved = vars(saved) if isinstance(saved, argparse.Namespace) else saved
    parameters = inspect.signature(ThirdOnlyOffsetFullCrossAttentionCLSGateFusionModel).parameters
    options = {key: value for key, value in saved.items() if key in parameters}
    options.update(overrides)
    model = ThirdOnlyOffsetFullCrossAttentionCLSGateFusionModel(**options)
    model.load_state_dict(state, strict=True, assign=True)
    remaining = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if remaining:
        raise RuntimeError(f'Unmaterialized model tensors: {remaining}')
    model.requires_grad_(False).eval().to(device)
    return model
