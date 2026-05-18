---
title: Gaussianoptics 四组元变焦镜头自动设计
emoji: 🔭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: Auto-design 4-group zoom lens
---

# Gaussianoptics — 四组元变焦镜头初始结构自动设计

> ⚠️ **WIP / research-grade**: This project is under active research and development. Optimization quality, parameter defaults, and UI are subject to change. Results are intended as **starting points for further refinement in Zemax / CODE V**, not as production-ready lens designs.

## 这是什么

基于薄透镜近似（Gaussian optics）和差分进化优化（DE）的四组元变焦镜头自动设计工具。输入光学规格 → 自动生成 G1/G2/G3/G4 四组焦距、变焦位置轨迹、各组通光口径。

## 怎么用

1. 在左侧填入光学规格（默认值是一个 12× zoom 标准用例）
2. 选择优化轮数（1 / 3 / 6 轮）
3. 点击「🚀 开始优化」
4. 等待结果（云端 2 vCPU 单轮约 2-3 分钟）
5. 右侧查看变焦曲线分析图 + 下载 CSV

## 输出

- **变焦曲线分析图**：4 子图（凸轮轨迹 / EFL / G3 放大率 / 各组通光口径）
- **CSV 文件**：5 个关键变焦位置（W / MW / M / MT / T）的间距、焦距、口径数据

## 性能说明

云端 2 vCPU 单轮约 4-5 分钟：

| 档位 | 预期耗时 |
|---|---|
| 1 轮（快速）| ~4-5 分钟  |
| 3 轮（标准）| ~13-15 分钟 |
| 6 轮（精细）| ~25-30 分钟 |

研究级别使用建议 fork 仓库本地运行（多核加速，质量更稳定）。

## 仓库

- 主仓库：https://github.com/yuboswag/Gaussianoptics
- 配套实际透镜生成（gauss_to_lens）：https://github.com/yuboswag/GausstoActuallens

## License

MIT —— see [LICENSE](LICENSE) for details.