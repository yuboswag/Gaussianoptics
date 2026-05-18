"""Gaussianoptics 四组元变焦镜头自动设计 Web UI."""

import os
os.environ.setdefault('GAUSSIAN_DE_WORKERS', '2')

import csv
import queue
import tempfile
import threading
import time

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 中文字体（Linux 容器需装 fonts-noto-cjk，本机用系统已有字体）
import numpy as np

from optimizer import ZoomLensOptimizer
from simulator import ZoomConfig


# ==================== 默认值 ====================
# 主面板字段：取自用户截图（11.9/121/110/8/55/60/7.6/4/5.6/40/61）
# 高级字段：我的猜测，需要用户验证
DEFAULTS = {
    # 主面板
    'f_wide': 11.9, 'f_tele': 121.0,
    'ttl_target': 110.0, 'bfd_target': 8.0,
    'f1': 55.0, 'f4': 60.0,
    'sensor_size': 7.6,
    'f_number': 4.0, 'f_number_tele': 5.6,
    'max_ca1': 40.0, 'num_positions': 61,
    # 高级（折叠）
    'g1_thickness': 15.0, 'g2_thickness': 9.0,
    'g3_thickness': 7.0, 'g4_thickness': 21.0,
    'n_eff_G1': 1.65, 'n_eff_G2': 1.70,
    'n_eff_G3': 1.65, 'n_eff_G4': 1.65,
    'v_eff_G1': 50.0, 'v_eff_G2': 30.0,
    'v_eff_G3': 50.0, 'v_eff_G4': 50.0,
    'stop_shift': 0.0, 'stop_group': 3,
    'vignetting': 0.7, 'constant_f_number': False,
}


# ==================== 绘图（搬自 gui.py:786-825）====================
def plot_trajectory(traj, sys_obj, cfg):
    fig = plt.Figure(figsize=(10, 8))
    fig.suptitle('4-Group Zoom Lens Kinematics & Aperture Analysis', fontsize=13)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    N = len(traj['efl'])
    x = np.arange(N)

    # 凸轮曲线
    if 'root1_z3' in traj:
        ax1.plot(x, traj['root1_z3'], '.', color='#CCCCCC', markersize=2)
        ax1.plot(x, traj['root2_z3'], '.', color='#AAAAAA', markersize=2)
    ax1.plot(x, traj['z3'], 'r-', linewidth=2, label='G3 (Compensator)')
    ax1.plot(x, traj['z2'], 'b--', linewidth=2, label='G2 (Variator)')
    ax1.axhline(sys_obj.z_G1, color='k', linestyle=':', label='G1')
    ax1.axhline(sys_obj.z_G4_ref, color='g', linestyle=':', label='G4')
    ax1.set_title('Cam Curve Trajectories')
    ax1.set_ylabel('Lens position z (mm)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # EFL
    ax2.plot(x, traj['efl'], 'g-', linewidth=2)
    ax2.axhline(cfg.f_wide, color='r', linestyle='--')
    ax2.axhline(cfg.f_tele, color='b', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_title('System EFL Verification')
    ax2.grid(True, which='both', alpha=0.3)

    # G3 放大率
    ax3.plot(x, traj['m3'], 'm-', linewidth=2)
    ax3.axhline(-1.0, color='r', linestyle='--')
    ax3.set_title('G3 Magnification m3')
    ax3.grid(True, alpha=0.3)

    # 通光口径
    ax4.plot(x, traj['CA1'], 'k-', label='CA1')
    ax4.plot(x, traj['CA2'], 'b-', label='CA2')
    ax4.plot(x, traj['CA3'], 'r-', label='CA3')
    ax4.plot(x, traj['CA4'], 'g-', label='CA4')
    max_ca1_val = float(np.max(traj['CA1']))
    ax4.set_title(f'Group Aperture Requirements (G1 Max: {max_ca1_val:.1f}mm)')
    ax4.set_ylabel('Diameter CA (mm)')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ==================== CSV 导出（5 个关键位置）====================
def export_csv(traj, cfg):
    N = len(traj['efl'])
    indices = [
        ('广角 W', 0),
        ('中广 MW', N // 4),
        ('中焦 M', N // 2),
        ('中长 MT', 3 * N // 4),
        ('长焦 T', N - 1),
    ]
    headers = ['位置', '焦距 EFL (mm)', 'd1 (G1-G2) (mm)', 'd2 (G2-G3) (mm)',
               'd3 (G3-G4) (mm)', 'm3', 'CA1 (mm)', 'CA2 (mm)', 'CA3 (mm)', 'CA4 (mm)']
    rows = []
    for name, i in indices:
        rows.append({
            '位置': name,
            '焦距 EFL (mm)': f"{traj['efl'][i]:.3f}",
            'd1 (G1-G2) (mm)': f"{traj['d1'][i]:.3f}",
            'd2 (G2-G3) (mm)': f"{traj['d2'][i]:.3f}",
            'd3 (G3-G4) (mm)': f"{traj['d3'][i]:.3f}",
            'm3': f"{traj['m3'][i]:.3f}",
            'CA1 (mm)': f"{traj['CA1'][i]:.3f}",
            'CA2 (mm)': f"{traj['CA2'][i]:.3f}",
            'CA3 (mm)': f"{traj['CA3'][i]:.3f}",
            'CA4 (mm)': f"{traj['CA4'][i]:.3f}",
        })
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False,
        encoding='utf-8-sig', newline='',
    )
    writer = csv.DictWriter(tmp, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()
    return tmp.name


# ==================== 主优化流程（流式 generator）====================
def run_optimization(
    f_wide, f_tele, ttl_target, bfd_target, f1, f4,
    sensor_size, f_number, f_number_tele, max_ca1, num_positions,
    g1_t, g2_t, g3_t, g4_t,
    n_g1, n_g2, n_g3, n_g4,
    v_g1, v_g2, v_g3, v_g4,
    stop_shift, stop_group, vignetting, constant_f,
    n_rounds,
):
    log_lines = []
    def log(msg):
        log_lines.append(msg)

    yield "⏳ 准备配置...", None, None

    try:
        cfg = ZoomConfig(
            f_wide=float(f_wide), f_tele=float(f_tele),
            ttl_target=float(ttl_target), bfd_target=float(bfd_target),
            f1=float(f1), f4=float(f4),
            sensor_size=float(sensor_size),
            f_number=float(f_number), f_number_tele=float(f_number_tele),
            max_ca1=float(max_ca1), num_positions=int(num_positions),
            stop_shift=float(stop_shift), stop_group=int(stop_group),
            vignetting=float(vignetting), constant_f_number=bool(constant_f),
            g1_thickness=float(g1_t), g2_thickness=float(g2_t),
            g3_thickness=float(g3_t), g4_thickness=float(g4_t),
            n_eff_G1=float(n_g1), n_eff_G2=float(n_g2),
            n_eff_G3=float(n_g3), n_eff_G4=float(n_g4),
            v_eff_G1=float(v_g1), v_eff_G2=float(v_g2),
            v_eff_G3=float(v_g3), v_eff_G4=float(v_g4),
        )
    except Exception as e:
        yield f"❌ 配置错误: {e}", None, None
        return

    n_rounds = int(n_rounds)
    log(f">>> 启动工程级光学寻优 (共 {n_rounds} 轮)")
    log(f"    目标: {cfg.f_wide}-{cfg.f_tele}mm | TTL: {cfg.ttl_target}mm")
    yield "\n".join(log_lines), None, None

    best_optimizer = None
    best_score = float('inf')
    t_start = time.perf_counter()

    for i in range(1, n_rounds + 1):
        log(f"\n--- [第 {i}/{n_rounds} 轮] 正在计算... ---")
        yield "\n".join(log_lines), None, None

        # 同步调用 optimize（在 generator 主上下文，避免 daemon thread 干扰 DE 子进程）
        # 代价：单轮内看不到中间 callback，但减少 ~3× 慢化
        try:
            opt = ZoomLensOptimizer(cfg)
            extra_seeds = [i * 100 + j for j in range(1, 4)] if i > 1 else None
            opt.optimize(callback=None, extra_seeds=extra_seeds)
            current_opt = opt
        except Exception as e:
            log(f"    ✗ 异常: {e}")
            yield "\n".join(log_lines), None, None
            continue
        if current_opt and current_opt.best_params is not None:
            breakdown = current_opt.get_penalty_diagnostics(current_opt.best_params)
            current_loss = breakdown.get("总分 (Total)", float('inf'))
            line = f"    ✓ 得分: {current_loss:.2e}"
            if current_loss < best_score:
                best_score = current_loss
                best_optimizer = current_opt
                line += " (👑 新纪录)"
            log(line)
        else:
            log(f"    ✗ 未收敛")
        yield "\n".join(log_lines), None, None

    elapsed = time.perf_counter() - t_start

    if best_optimizer is None:
        log(f"\n❌ {n_rounds} 轮全部未收敛，请检查参数。")
        yield "\n".join(log_lines), None, None
        return

    log(f"\n🏆 最终结果 (最低得分: {best_score:.2e}, 总耗时 {elapsed:.1f}s)")
    breakdown = best_optimizer.get_penalty_diagnostics(best_optimizer.best_params)
    total_loss = breakdown.get("总分 (Total)", 1e-9)
    log("📊 【优化结果深度诊断】")
    for key, val in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        if key == "总分 (Total)":
            continue
        pct = (val / total_loss) * 100 if total_loss > 0 else 0
        if val > 1.0:
            tag = "🔴" if pct > 10 else "  "
            log(f"  {tag} {key}: {val:.2e} (占比 {pct:.1f}%)")

    try:
        traj = best_optimizer.best_trajectory
        sys_obj = best_optimizer.system
        fig = plot_trajectory(traj, sys_obj, cfg)
        csv_path = export_csv(traj, cfg)
        log("\n✓ 曲线图和 CSV 已生成。")
        yield "\n".join(log_lines), fig, csv_path
    except Exception as e:
        log(f"⚠ 后处理失败: {e}")
        yield "\n".join(log_lines), None, None


# ==================== Gradio UI ====================
# ==================== Gradio UI ====================
def build_ui():
    with gr.Blocks(title="Gaussianoptics 变焦镜头自动设计") as demo:
        gr.Markdown(
            "# 四组元变焦镜头自动设计 (Gaussianoptics)\n\n"
            "**中文**：基于薄透镜近似 + 差分进化优化，自动生成变焦镜头的高斯解初始结构参数与凸轮曲线。"
            "输入光学规格 → 选择优化轮数 → 点击 **🚀 开始优化** → 等待结果。\n\n"
            "**English**: Auto-generate Gaussian-optics initial structure parameters and cam curves for zoom lenses "
            "using thin-lens approximation + differential evolution. "
            "Enter optical specs → select number of optimization rounds → click **🚀 Start Optimization** → wait for results."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 基本光学参数")
                with gr.Row():
                    f_wide = gr.Number(label="广角焦距 f_wide (mm)", value=DEFAULTS['f_wide'])
                    f_tele = gr.Number(label="长焦焦距 f_tele (mm)", value=DEFAULTS['f_tele'])
                with gr.Row():
                    ttl_target = gr.Number(label="总长 TTL (mm)", value=DEFAULTS['ttl_target'])
                    bfd_target = gr.Number(label="后焦距 BFD (mm)", value=DEFAULTS['bfd_target'])
                with gr.Row():
                    f1 = gr.Number(label="前组焦距 f1 (mm)", value=DEFAULTS['f1'])
                    f4 = gr.Number(label="后组焦距 f4 (mm)", value=DEFAULTS['f4'])
                with gr.Row():
                    sensor_size = gr.Number(label="像面直径 (mm)", value=DEFAULTS['sensor_size'])
                    num_positions = gr.Number(label="采样点数 N", value=DEFAULTS['num_positions'], precision=0)
                with gr.Row():
                    f_number = gr.Number(label="广角 F# (W)", value=DEFAULTS['f_number'])
                    f_number_tele = gr.Number(label="长焦 F# (T)", value=DEFAULTS['f_number_tele'])
                max_ca1 = gr.Number(label="G1 最大宽度 CA1 (mm)", value=DEFAULTS['max_ca1'])

                with gr.Accordion("高级参数（点击展开）", open=False):
                    gr.Markdown("**组元厚度 (mm)**")
                    with gr.Row():
                        g1_t = gr.Number(label="t_G1", value=DEFAULTS['g1_thickness'])
                        g2_t = gr.Number(label="t_G2", value=DEFAULTS['g2_thickness'])
                        g3_t = gr.Number(label="t_G3", value=DEFAULTS['g3_thickness'])
                        g4_t = gr.Number(label="t_G4", value=DEFAULTS['g4_thickness'])
                    gr.Markdown("**等效折射率 n_eff**")
                    with gr.Row():
                        n_g1 = gr.Number(label="G1", value=DEFAULTS['n_eff_G1'])
                        n_g2 = gr.Number(label="G2", value=DEFAULTS['n_eff_G2'])
                        n_g3 = gr.Number(label="G3", value=DEFAULTS['n_eff_G3'])
                        n_g4 = gr.Number(label="G4", value=DEFAULTS['n_eff_G4'])
                    gr.Markdown("**等效阿贝数 v_eff**")
                    with gr.Row():
                        v_g1 = gr.Number(label="G1", value=DEFAULTS['v_eff_G1'])
                        v_g2 = gr.Number(label="G2", value=DEFAULTS['v_eff_G2'])
                        v_g3 = gr.Number(label="G3", value=DEFAULTS['v_eff_G3'])
                        v_g4 = gr.Number(label="G4", value=DEFAULTS['v_eff_G4'])
                    gr.Markdown("**光阑与渐晕**")
                    with gr.Row():
                        stop_shift = gr.Number(label="光阑偏移 (mm)", value=DEFAULTS['stop_shift'])
                        stop_group = gr.Number(label="光阑组号 (1-4)", value=DEFAULTS['stop_group'], precision=0)
                    with gr.Row():
                        vignetting = gr.Slider(label="渐晕系数", minimum=0.0, maximum=1.0,
                                               step=0.05, value=DEFAULTS['vignetting'])
                        constant_f = gr.Checkbox(label="恒定 F#", value=DEFAULTS['constant_f_number'])

                gr.Markdown("### 优化设置")
                n_rounds = gr.Radio(
                    choices=[
                        ("快速（1 轮，约 4-5 分钟）", 1),
                        ("标准（3 轮，约 12-15 分钟）", 3),
                        ("精细（6 轮,  约 25-30 分钟）", 6),
                    ],
                    value=1,
                    label="优化轮数",
                )
                run_btn = gr.Button("🚀 开始优化", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 优化日志")
                log_output = gr.Textbox(label="", lines=20, max_lines=30, interactive=False)
                gr.Markdown("### 变焦曲线分析")
                plot_output = gr.Plot(label="")
                csv_output = gr.File(label="下载结果 CSV", interactive=False)

        run_btn.click(
            fn=run_optimization,
            inputs=[
                f_wide, f_tele, ttl_target, bfd_target, f1, f4,
                sensor_size, f_number, f_number_tele, max_ca1, num_positions,
                g1_t, g2_t, g3_t, g4_t,
                n_g1, n_g2, n_g3, n_g4,
                v_g1, v_g2, v_g3, v_g4,
                stop_shift, stop_group, vignetting, constant_f,
                n_rounds,
            ],
            outputs=[log_output, plot_output, csv_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(ssr_mode=False)