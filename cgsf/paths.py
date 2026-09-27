"""Repository-relative paths, confined to the standalone code directory."""
from pathlib import Path

# Internal anchor only: no machine-specific path is stored in configuration.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = Path("checkpoints/checkpoint-epoch")
DEFAULT_DATA_DIR = Path("data/test_new")
DEFAULT_GT_DIR = DEFAULT_DATA_DIR / "GT_ISTD"
DEFAULT_IMAGE_DIR = Path("data/images")
DEFAULT_TEST_OUTPUT = Path("test_results")
DEFAULT_INFER_OUTPUT = Path("predictions")
DEFAULT_DINO_CONFIG = Path("cgsf/dinov3_config.json")


def ensure_local(path):
    """Validate an internal path, including existing symlink targets."""
    path = Path(path)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"路径必须位于 gitcgsf 文件夹内: {path}")
    return resolved


def resolve_relative(path):
    """Resolve a user path against ROOT, never against the working directory."""
    path = Path(path)
    if path.is_absolute():
        raise ValueError(f"请使用相对于 gitcgsf 的路径，不接受绝对路径: {path}")
    return ensure_local(path)
