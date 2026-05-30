"""Batch low-light image enhancement for HW02.

The script implements a lightweight traditional CV pipeline:
adaptive gamma, conservative denoising, LIME-style illumination correction,
and CLAHE. It also produces a simple gamma+CLAHE baseline, comparison images,
and no-reference metrics for the report.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class EnhancementConfig:
    gamma_min: float = 0.52
    gamma_max: float = 0.88
    baseline_gamma: float = 0.62
    clahe_clip_limit: float = 1.25
    clahe_tile_grid_size: int = 8
    illumination_blur_sigma: float = 24.0
    illumination_strength: float = 0.18
    denoise_strength: float = 3.0
    saturation_gain: float = 1.0
    comparison_width: int = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance low-light images in batch.")
    parser.add_argument(
        "--input",
        default="release/inputs for coding project/LowIllumination",
        type=Path,
        help="Directory containing low-light images.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        type=Path,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising for quick ablation.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda p: natural_key(p.name))


def natural_key(text: str) -> list[object]:
    parts: list[object] = []
    buf = ""
    is_digit = False
    for char in text:
        char_is_digit = char.isdigit()
        if buf and char_is_digit != is_digit:
            parts.append(int(buf) if is_digit else buf.lower())
            buf = ""
        buf += char
        is_digit = char_is_digit
    if buf:
        parts.append(int(buf) if is_digit else buf.lower())
    return parts


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise ValueError(f"Failed to write image: {path}")


def to_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def to_uint8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(rgb_float * 255.0, 0, 255).round().astype(np.uint8)


def luminance(rgb_float: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb_float[:, :, 0]
        + 0.7152 * rgb_float[:, :, 1]
        + 0.0722 * rgb_float[:, :, 2]
    )


def gamma_correct(rgb_float: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(rgb_float, 0.0, 1.0), gamma)


def adaptive_gamma(rgb_float: np.ndarray, cfg: EnhancementConfig) -> float:
    mean_luma = float(luminance(rgb_float).mean())
    # Map very dark images near gamma_min and brighter low-light images near gamma_max.
    gamma = cfg.gamma_min + (cfg.gamma_max - cfg.gamma_min) * np.clip(mean_luma / 0.45, 0.0, 1.0)
    return float(np.clip(gamma, cfg.gamma_min, cfg.gamma_max))


def apply_clahe_l_channel(rgb: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_grid_size, cfg.clahe_tile_grid_size),
    )
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def denoise_rgb(rgb: np.ndarray, cfg: EnhancementConfig, enabled: bool) -> np.ndarray:
    if not enabled or cfg.denoise_strength <= 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=cfg.denoise_strength,
        hColor=cfg.denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)


def illumination_correct(rgb_float: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    initial_map = np.max(rgb_float, axis=2)
    illumination = cv2.GaussianBlur(
        initial_map,
        ksize=(0, 0),
        sigmaX=cfg.illumination_blur_sigma,
        sigmaY=cfg.illumination_blur_sigma,
    )
    illumination = np.clip(illumination, 0.18, 1.0)
    corrected = rgb_float / illumination[:, :, np.newaxis]
    corrected = np.clip(corrected, 0.0, 1.0)
    return (1.0 - cfg.illumination_strength) * rgb_float + cfg.illumination_strength * corrected


def adjust_saturation(rgb: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg.saturation_gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def enhance_baseline(rgb: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    gamma_rgb = to_uint8(gamma_correct(to_float(rgb), cfg.baseline_gamma))
    return apply_clahe_l_channel(gamma_rgb, cfg)


def enhance_main(rgb: np.ndarray, cfg: EnhancementConfig, denoise_enabled: bool = True) -> tuple[np.ndarray, float]:
    rgb_float = to_float(rgb)
    gamma = adaptive_gamma(rgb_float, cfg)
    gamma_rgb = to_uint8(gamma_correct(rgb_float, gamma))
    denoised_rgb = denoise_rgb(gamma_rgb, cfg, denoise_enabled)
    corrected_float = illumination_correct(to_float(denoised_rgb), cfg)
    corrected_rgb = to_uint8(corrected_float)
    clahe_rgb = apply_clahe_l_channel(corrected_rgb, cfg)
    saturated_rgb = adjust_saturation(clahe_rgb, cfg)
    return saturated_rgb, gamma


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def image_metrics(image_name: str, method: str, rgb: np.ndarray) -> dict[str, object]:
    rgb_float = to_float(rgb)
    y = to_uint8(luminance(rgb_float))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "image": image_name,
        "method": method,
        "mean_luminance": f"{float(y.mean()):.3f}",
        "std_luminance": f"{float(y.std()):.3f}",
        "entropy": f"{entropy(y):.4f}",
        "mean_saturation": f"{float(hsv[:, :, 1].mean()):.3f}",
    }


def resize_for_comparison(rgb: np.ndarray, width: int) -> np.ndarray:
    scale = width / rgb.shape[1]
    height = max(1, int(round(rgb.shape[0] * scale)))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def add_label(rgb: np.ndarray, label: str) -> np.ndarray:
    label_height = 34
    canvas = np.full((rgb.shape[0] + label_height, rgb.shape[1], 3), 245, dtype=np.uint8)
    canvas[label_height:, :, :] = rgb
    cv2.putText(
        canvas,
        label,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_comparison(original: np.ndarray, baseline: np.ndarray, enhanced: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
    panels = [
        add_label(resize_for_comparison(original, cfg.comparison_width), "Original"),
        add_label(resize_for_comparison(baseline, cfg.comparison_width), "Baseline: gamma + CLAHE"),
        add_label(resize_for_comparison(enhanced, cfg.comparison_width), "Main method"),
    ]
    target_height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < target_height:
            pad = np.full((target_height - panel.shape[0], panel.shape[1], 3), 245, dtype=np.uint8)
            panel = np.vstack((panel, pad))
        padded.append(panel)
    separator = np.full((target_height, 10, 3), 255, dtype=np.uint8)
    return np.hstack([padded[0], separator, padded[1], separator, padded[2]])


def clean_output_dirs(output_dir: Path) -> None:
    for name in ("baseline", "enhanced", "comparisons"):
        directory = output_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                path.unlink()


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "method", "mean_luminance", "std_luminance", "entropy", "mean_saturation"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_all(input_dir: Path, output_dir: Path, cfg: EnhancementConfig, denoise_enabled: bool) -> None:
    images = list_images(input_dir)
    if not images:
        raise ValueError(f"No supported images found in: {input_dir}")

    clean_output_dirs(output_dir)
    metric_rows: list[dict[str, object]] = []

    for image_path in images:
        original = read_rgb(image_path)
        baseline = enhance_baseline(original, cfg)
        enhanced, gamma = enhance_main(original, cfg, denoise_enabled=denoise_enabled)
        comparison = make_comparison(original, baseline, enhanced, cfg)

        stem = image_path.stem
        write_rgb(output_dir / "baseline" / f"{stem}_baseline.png", baseline)
        write_rgb(output_dir / "enhanced" / f"{stem}_enhanced.png", enhanced)
        write_rgb(output_dir / "comparisons" / f"{stem}_comparison.png", comparison)

        metric_rows.extend(
            [
                image_metrics(image_path.name, "original", original),
                image_metrics(image_path.name, "baseline", baseline),
                image_metrics(image_path.name, "enhanced", enhanced),
            ]
        )
        print(f"processed {image_path.name}: adaptive_gamma={gamma:.3f}")

    write_metrics(output_dir / "metrics.csv", metric_rows)
    print(f"done: {len(images)} images, metrics written to {output_dir / 'metrics.csv'}")


def main() -> None:
    args = parse_args()
    cfg = EnhancementConfig()
    process_all(args.input, args.output, cfg, denoise_enabled=not args.no_denoise)


if __name__ == "__main__":
    main()
