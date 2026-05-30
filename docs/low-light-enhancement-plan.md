# Low Illumination Image Enhancement Plan

## Summary

本项目选择作业 2 的第三题：低照度图像增强。目标是在短时间内完成一个可运行、可解释、效果明显的传统计算摄影方案，并生成报告所需的增强结果、前后对比图和简单无参考指标。

主方案采用稳妥省时路线：

1. Adaptive gamma correction 提升整体亮度。
2. Light denoise 抑制低照度提亮后的噪声。
3. Retinex/LIME-style illumination correction 改善不均匀照明。
4. CLAHE 增强局部对比度。
5. Color and exposure clipping 保持自然颜色并避免过曝。

不使用深度学习训练作为主方案，不追求复杂优化版 LIME，只实现适合作业提交的传统 CV 流水线。

## RALPLAN-DR

### Principles

- 短时间达标优先：代码能批量运行、结果能肉眼看出提升、报告能解释清楚。
- 自己实现算法主线：可以使用 OpenCV/NumPy 的基础图像操作，但不直接套完整深度学习模型。
- 输出可复现：输入、参数、输出目录和指标表固定。
- 评价诚实：没有 ground truth，不使用 PSNR/SSIM 作为主指标。
- 文档和实现双向校验：方案变更要更新实现，代码行为变更要更新方案。

### Decision Drivers

- 给定数据是 10 张独立低照度 BMP，适合单图增强，不需要图像配准或曝光时间。
- 低照度增强最容易产生稳定的 before/after 可视化结果。
- 传统方法参数少、依赖轻、报告叙述空间足够。

### Viable Options

| Option | Pipeline | Pros | Cons |
| --- | --- | --- | --- |
| A. Minimal baseline | gamma correction + CLAHE | 实现最快，适合兜底 | 算法深度较弱，容易过曝或放大噪声 |
| B. Recommended | adaptive gamma + denoise + Retinex/LIME-style illumination + CLAHE | 效果、报告和实现复杂度平衡最好 | 需要少量参数调试 |
| C. Multi-method comparison | baseline + Retinex + LIME-like + ablation | 报告更丰满 | 时间更长，调参和图表更多 |

### Rejected Alternatives

- 深度学习增强模型：效果可能好，但不符合短时间、轻依赖、自己实现主线的目标。
- 复杂 LIME 优化求解：论文味更足，但实现和调试成本高；本项目只采用其 illumination map 思想。
- 只做 histogram equalization：太粗糙，容易产生噪声、偏色和不自然对比。

## ADR

### Decision

采用 Option B，并保留 Option A 作为 baseline。最终提交围绕一个主脚本批量处理 `release/inputs for coding project/LowIllumination/*.bmp`，输出增强图、对比图和指标表。

### Status

Accepted for first implementation.

### Consequences

- 代码应优先清晰和可调参，而不是追求最先进指标。
- 报告可以围绕 Retinex 图像模型和 LIME illumination map 思想展开。
- 所有实验结论以视觉对比和无参考统计指标支持。

## Algorithm Design

### Input

- Source directory: `release/inputs for coding project/LowIllumination/`
- File pattern: `*.bmp`
- Expected count: 10 images

### Baseline

Baseline 用于报告对照，不作为最终主结果：

1. Convert RGB to LAB or HSV.
2. Apply fixed or adaptive gamma on luminance/value channel.
3. Apply CLAHE on luminance/value channel.
4. Convert back to RGB and save.

### Main Pipeline

Recommended default pipeline:

1. Load image as RGB float in `[0, 1]`.
2. Estimate image brightness from luminance channel.
3. Compute adaptive gamma:
   - darker image uses lower gamma to brighten more.
   - gamma should be clipped to a stable range, for example `[0.45, 0.85]`.
4. Apply gamma correction to luminance or RGB image.
5. Apply light denoise:
   - preferred: `cv2.fastNlMeansDenoisingColored` or bilateral filtering.
   - keep denoise conservative to avoid texture loss.
6. Estimate illumination map:
   - LIME-style initial map: max over RGB channels.
   - smooth/refine with Gaussian or bilateral filtering.
7. Correct illumination:
   - divide image by smoothed illumination map with epsilon.
   - blend corrected image with gamma result to avoid over-enhancement.
8. Apply CLAHE on LAB L channel.
9. Adjust saturation mildly if image becomes gray.
10. Clip to `[0, 255]`, convert to `uint8`, and save.

### Parameters To Expose

- `gamma_min`, `gamma_max`
- `clahe_clip_limit`
- `clahe_tile_grid_size`
- `illumination_blur_sigma`
- `illumination_strength`
- `denoise_strength`
- `saturation_gain`

Defaults should work for all 10 images without per-image manual tuning.

## Outputs

Expected generated paths:

- `outputs/enhanced/{stem}_enhanced.png`
- `outputs/baseline/{stem}_baseline.png`
- `outputs/comparisons/{stem}_comparison.png`
- `outputs/metrics.csv`

Comparison image should show at least:

- original
- baseline
- main enhanced result

`metrics.csv` should contain one row per image and method where practical.

Recommended columns:

- `image`
- `method`
- `mean_luminance`
- `std_luminance`
- `entropy`
- `mean_saturation`

## Report Direction

Report structure:

1. Task introduction: low illumination causes low brightness, low contrast, high noise.
2. Method: describe adaptive gamma, Retinex/LIME-style illumination estimation, CLAHE, denoising.
3. Implementation details: input directory, parameters, output generation.
4. Experiments: before/baseline/enhanced comparison images for all 10 inputs and metrics table.
5. Discussion: improvements, limitations, overexposure/noise tradeoff.
6. Conclusion: method improves visibility and local contrast with simple traditional CV operations.

Suggested references:

- LIME: Low-Light Image Enhancement via Illumination Map Estimation, arXiv:1605.05034.
- Retinex-based low-light enhancement literature for illumination-reflectance model.
- OpenCV CLAHE documentation for local contrast enhancement.

## Test Plan

Minimum verification:

1. Run the processing script on all BMP files in `release/inputs for coding project/LowIllumination/`.
2. Confirm output directories exist and contain expected image counts.
3. Open several comparison images and visually check:
   - dark regions are brighter.
   - scene details become more visible.
   - highlights are not badly clipped.
   - colors remain reasonably natural.
   - noise is not aggressively amplified.
4. Confirm `outputs/metrics.csv` is readable and includes every input image.

Suggested smoke command after implementation:

```powershell
python src/enhance_low_light.py --input "release/inputs for coding project/LowIllumination" --output outputs
```

## Assumptions

- Python environment has `cv2`, `numpy`, and `PIL` available.
- No normal-light ground truth is provided.
- Submission values visual quality and method explanation more than benchmark accuracy.
- A single global default parameter set is acceptable for all 10 images.

## Boundaries

In scope:

- Traditional CV enhancement implementation.
- Batch output generation.
- Baseline comparison.
- Simple metrics for report.

Out of scope for the first version:

- Deep learning training or pretrained model inference.
- Per-image manual tuning.
- Full LIME optimization solver.
- PSNR/SSIM claims without ground truth.
- HDR or deblurring task implementation.

## Report Figure Policy

The report should include all 10 comparison results, not only selected examples. To keep the report readable:

- Put all `outputs/comparisons/*_comparison.png` images in the results section or a clearly labeled results appendix.
- Prefer a compact but legible layout, such as two figure pages with multiple comparison images, or one image per figure if page count is not a concern.
- Keep the metrics table full-dataset as well, covering all 10 inputs and the original/baseline/enhanced methods.
- In the discussion, mention both successful examples and difficult cases such as high-noise or dense-texture images.

## Next Lane

Next recommended step: implement the batch processing script and generate first-pass outputs, then visually inspect comparison images and adjust default parameters once.
