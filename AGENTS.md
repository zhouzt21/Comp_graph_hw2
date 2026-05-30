# Project AGENTS.md

## Language

- 默认用中文沟通；代码、命令、路径、API 名称保留英文。
- 回答保持简洁，优先给可执行结论和必要背景。

## Project Scope

- 本项目当前作业选题固定为 `Low Illumination Image Enhancement`。
- 方案源头文件是 `docs/low-light-enhancement-plan.md`。
- 后续实现应围绕低照度增强，不主动切换到 HDR、Deblurring 或深度学习训练方案，除非用户明确要求。

## Plan And Implementation Sync

- 修改实现前，先阅读 `docs/low-light-enhancement-plan.md`，确认算法流程、输出目录和验收标准。
- 修改方案文档时，同步检查实现计划是否仍匹配；如果暂不实现，必须在文档中标出待办或边界。
- 修改代码、脚本、报告素材后，反向检查方案文档是否过期；若行为、参数、输出位置或验收方式变化，需要同步更新文档。
- 最终交付前做双向校验：
  - 文档中承诺的输入、输出、指标、验证步骤在代码中存在。
  - 代码实际生成的文件、参数和限制在文档中有说明。

## Implementation Rules

- 优先使用 Python + OpenCV + NumPy + Pillow；不要引入深度学习框架作为主依赖。
- 算法主线保持稳妥省时：adaptive gamma、denoise、Retinex/LIME-style illumination correction、CLAHE。
- 保留一个简单 baseline，便于报告中展示主方法的改进。
- 输出应可复现、可批处理，不依赖手工逐图操作。
- 不做无关重构，不移动 `release/` 中的原始作业资料。

## Expected Outputs

- 增强图：`outputs/enhanced/`
- 前后对比图：`outputs/comparisons/`
- 指标表：`outputs/metrics.csv`
- 可选报告素材或草稿：`report/`

## Verification

- 最小验证是运行批处理脚本处理全部 `release/inputs for coding project/LowIllumination/*.bmp`。
- 验证输出数量、文件可读性、对比图是否生成、`metrics.csv` 是否包含每张输入图。
- 评价指标使用无参考指标，例如平均亮度、亮度标准差/对比度、熵；没有 ground truth 时不要声称 PSNR/SSIM。
- 视觉检查至少覆盖暗场、局部强光、室外夜景三类样例，避免明显过曝、严重偏色和噪声爆炸。

## Safety

- 不读取、输出、复制或记录 secrets、API key、token、密码。
- 不主动修改 `.env`、凭据文件或私密配置。
- 不使用破坏性 git 命令，例如 `git reset --hard`、强制覆盖、删除未确认文件。
