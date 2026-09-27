"""Predict probability maps and binary masks for an image or image directory."""
import argparse

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from cgsf.checkpoint import DEFAULT_CHECKPOINT, load_model
from cgsf.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from cgsf.paths import DEFAULT_IMAGE_DIR, DEFAULT_INFER_OUTPUT, resolve_relative, ensure_local


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--output_dir', default=str(DEFAULT_INFER_OUTPUT))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be in [0, 1]')
    try:
        source = resolve_relative(args.input)
        output = resolve_relative(args.output_dir)
        resolve_relative(args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}) if source.is_dir() else [source]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No input images found')
    try:
        for path in paths:
            ensure_local(path)
    except ValueError as exc:
        parser.error(str(exc))
    model = load_model(args.checkpoint, args.device)
    device = torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with Image.open(path) as image:
            original_size = image.size
            image = TF.resize(image.convert('RGB'), [model.img_size] * 2, interpolation=InterpolationMode.BILINEAR)
            raw = TF.to_tensor(image).unsqueeze(0).to(device)
        normalized = TF.normalize(raw, IMAGENET_MEAN, IMAGENET_STD)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            prediction = model(normalized, image_raw=raw)
        prediction = torch.nn.functional.interpolate(prediction.float(), size=original_size[::-1], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        # Include the input extension to avoid collisions between a.jpg and a.png.
        np.save(ensure_local(output / f'{path.name}.prob.npy'), prediction)
        Image.fromarray((prediction * 255).round().astype(np.uint8)).save(ensure_local(output / f'{path.name}.prob.png'))
        Image.fromarray((prediction > args.threshold).astype(np.uint8) * 255).save(ensure_local(output / f'{path.name}.mask.png'))
        print(f'Predicted {path.name}')


if __name__ == '__main__':
    main()
