"""Dataset evaluation using the original image preprocessing and pixel metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cgsf.checkpoint import DEFAULT_CHECKPOINT, load_model
from cgsf.evaluation import test_on_subset
from cgsf.paths import DEFAULT_DATA_DIR, DEFAULT_GT_DIR, DEFAULT_TEST_OUTPUT, resolve_relative, ensure_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--data_dir', default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--gt_dir', default=str(DEFAULT_GT_DIR))
    parser.add_argument('--subsets', nargs='+', default=['test_BM', 'test_DC', 'test_DSC', 'test_SP', 'test_TG'])
    parser.add_argument('--output_dir', default=str(DEFAULT_TEST_OUTPUT))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--compute_auc', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--auc_sample_rate', type=float, default=1.0)
    parser.add_argument('--maxf1_num_bins', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or not 0 < args.auc_sample_rate <= 1 or args.maxf1_num_bins < 2:
        parser.error('Invalid batch size, worker count, AUC sample rate or histogram bins')
    try:
        resolve_relative(args.checkpoint)
        data_dir = resolve_relative(args.data_dir)
        gt_dir = resolve_relative(args.gt_dir)
        output = resolve_relative(args.output_dir)
        for subset in args.subsets:
            if Path(subset).name != subset or subset in {".", ".."}:
                raise ValueError("subsets 必须是直接子目录名")
            ensure_local(data_dir / subset)
    except ValueError as exc:
        parser.error(str(exc))
    for subset in args.subsets:
        if not (data_dir / subset).is_dir():
            parser.error(f'Test subset does not exist: {Path(args.data_dir) / subset}')
    if not gt_dir.is_dir():
        parser.error(f'GT directory does not exist: {args.gt_dir}')
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    print(f'Strict checkpoint load succeeded; device={device}, image size={model.img_size}')
    rng = np.random.default_rng(args.seed)
    results = {'checkpoint': args.checkpoint, 'img_size': model.img_size, 'options': vars(args), 'subsets': {}}
    for subset in args.subsets:
        metrics, count = test_on_subset(
            model, str(data_dir / subset), str(gt_dir), subset, device,
            args.batch_size, args.num_workers, model.img_size, args.compute_auc,
            args.auc_sample_rate, args.maxf1_num_bins, rng,
        )
        results['subsets'][subset] = {'num_samples': count, 'metrics': metrics}
        print(f'{subset}: {count} images\n{json.dumps(metrics, indent=2)}')
    output.mkdir(parents=True, exist_ok=True)
    path = ensure_local(output / 'evaluation_results.json')
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    print(f'Results saved to {path}')


if __name__ == '__main__':
    main()
