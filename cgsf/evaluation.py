import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD
from .paths import ensure_local
DEFAULT_TEST_SUBSETS = ["test_BM", "test_DC", "test_DSC", "test_SP", "test_TG"]


class EvalSubsetDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = str(ensure_local(self.image_paths[idx]))
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask_path = ensure_local(mask_path)

        image_pil = Image.open(image_path).convert("RGB")
        image_pil = TF.resize(
            image_pil,
            [self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        image_raw = TF.to_tensor(image_pil)
        image = TF.normalize(image_raw.clone(), mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if mask_path is None:
            mask = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        else:
            mask_pil = Image.open(mask_path).convert("L")
            mask_pil = TF.resize(
                mask_pil,
                [self.img_size, self.img_size],
                interpolation=InterpolationMode.NEAREST,
            )
            mask = TF.to_tensor(mask_pil)
            mask = (mask >= 0.5).float()

        return image, image_raw, mask, image_path


def cal_confusion_matrix(predict, target, threshold=0.5):
    predict = (predict > threshold).float()
    tp = torch.sum(predict * target, dim=(1, 2, 3))
    tn = torch.sum((1 - predict) * (1 - target), dim=(1, 2, 3))
    fp = torch.sum(predict * (1 - target), dim=(1, 2, 3))
    fn = torch.sum((1 - predict) * target, dim=(1, 2, 3))
    return tp, tn, fp, fn


def cal_metrics(tp, tn, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IoU": iou,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def update_pr_hist(pos_hist, neg_hist, preds, targets, num_bins=256):
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.uint8).reshape(-1)

    bin_idx = np.clip((preds * (num_bins - 1)).astype(np.int32), 0, num_bins - 1)

    pos_hist += np.bincount(
        bin_idx,
        weights=targets.astype(np.float64),
        minlength=num_bins,
    )
    neg_hist += np.bincount(
        bin_idx,
        weights=(1 - targets).astype(np.float64),
        minlength=num_bins,
    )


def cal_max_f1_from_hist(pos_hist, neg_hist):
    tp = np.cumsum(pos_hist[::-1])[::-1]
    fp = np.cumsum(neg_hist[::-1])[::-1]
    total_pos = pos_hist.sum()
    fn = total_pos - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = int(np.argmax(f1_scores))
    thresholds = np.linspace(0.0, 1.0, len(pos_hist), dtype=np.float32)
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def sample_for_auc(preds, targets, sample_rate, rng):
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.uint8).reshape(-1)

    if sample_rate >= 1.0 or preds.size == 0:
        return preds, targets

    sample_size = max(1, int(preds.size * sample_rate))
    if sample_size >= preds.size:
        return preds, targets

    indices = rng.choice(preds.size, size=sample_size, replace=False)
    return preds[indices], targets[indices]


def build_subset_dataset(subset_dir, gt_dir, img_size):
    subset_dir = ensure_local(subset_dir)
    gt_dir = ensure_local(gt_dir)
    images = []
    labels = []

    for test_file in sorted(os.listdir(subset_dir)):
        if not test_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        test_img_path = str(ensure_local(subset_dir / test_file))
        gt_img_path = str(ensure_local(gt_dir / (Path(test_file).stem + ".png")))

        images.append(test_img_path)
        if not os.path.isfile(gt_img_path):
            raise FileNotFoundError(f"缺少 GT，不能计算可靠指标: {gt_img_path}")
        labels.append(gt_img_path)

    if not images:
        raise ValueError(f"测试目录没有图像: {subset_dir}")
    dataset = EvalSubsetDataset(images, labels, img_size=img_size)
    return dataset, len(images)


def test_on_subset(
    model,
    subset_dir,
    gt_dir,
    subset_name,
    device,
    batch_size,
    num_workers,
    img_size,
    compute_auc=True,
    auc_sample_rate=0.1,
    maxf1_num_bins=256,
    rng=None,
):
    dataset_test, num_samples = build_subset_dataset(subset_dir, gt_dir, img_size=img_size)

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    model.eval()

    all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
    pos_hist = np.zeros(maxf1_num_bins, dtype=np.float64)
    neg_hist = np.zeros(maxf1_num_bins, dtype=np.float64)
    auc_preds = []
    auc_targets = []
    forward_start = time.perf_counter()

    if rng is None:
        rng = np.random.default_rng(42)

    with torch.inference_mode():
        for images_batch, images_raw_batch, masks, _ in tqdm(
            data_loader_test,
            desc=f"Testing {subset_name}",
            leave=False,
        ):
            images_batch = images_batch.to(device, non_blocking=True)
            images_raw_batch = images_raw_batch.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                predict = model(images_batch, image_raw=images_raw_batch)
            predict = predict.detach()

            tp, tn, fp, fn = cal_confusion_matrix(predict, masks)
            all_tp += tp.sum().item()
            all_tn += tn.sum().item()
            all_fp += fp.sum().item()
            all_fn += fn.sum().item()

            preds_np = predict.float().cpu().numpy().reshape(-1)
            targets_np = masks.float().cpu().numpy().reshape(-1)

            update_pr_hist(pos_hist, neg_hist, preds_np, targets_np, num_bins=maxf1_num_bins)

            if compute_auc:
                sampled_preds, sampled_targets = sample_for_auc(
                    preds_np, targets_np, auc_sample_rate, rng
                )
                auc_preds.append(sampled_preds)
                auc_targets.append(sampled_targets)

    forward_time = time.perf_counter() - forward_start

    metrics = cal_metrics(
        torch.tensor(all_tp, dtype=torch.float32),
        torch.tensor(all_tn, dtype=torch.float32),
        torch.tensor(all_fp, dtype=torch.float32),
        torch.tensor(all_fn, dtype=torch.float32),
    )
    metrics = {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
    metrics["eval_forward_time"] = float(forward_time)

    maxf1_start = time.perf_counter()
    max_f1, best_threshold = cal_max_f1_from_hist(pos_hist, neg_hist)
    metrics["MAX_F1"] = float(max_f1)
    metrics["best_threshold"] = float(best_threshold)
    metrics["eval_maxf1_time"] = float(time.perf_counter() - maxf1_start)

    if compute_auc and auc_preds:
        try:
            auc_preds = np.concatenate(auc_preds)
            auc_targets = np.concatenate(auc_targets)
            metrics["AUC_ROC"] = float(roc_auc_score(auc_targets, auc_preds))
            metrics["AP"] = float(average_precision_score(auc_targets, auc_preds))
            metrics["auc_sample_rate"] = float(auc_sample_rate)
            metrics["auc_num_pixels"] = int(auc_preds.size)
        except Exception as exc:
            print(f"Could not compute AUC/AP for {subset_name}: {exc}")

    return metrics, num_samples
