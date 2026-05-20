"""
gui.py
图形用户界面：参数输入、优化触发、结果展示、实时交互与数据导出
"""

import csv
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json    
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import ZoomConfig
from optimizer import ZoomLensOptimizer


class ZoomLensDesignerGUI:
    """正组补偿变焦镜头高斯设计工具（工程版）。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("正组补偿变焦镜头高斯设计 (工程版: 孔径估算与场曲控制)")
        # 【修改点 1】：增大默认窗口大小，解决显示不全的问题
        self.root.geometry("1500x950")

        self.optimizer: ZoomLensOptimizer | None = None
        self.params: dict[str, ttk.Entry] = {}
        self._after_id = None  

        self._create_widgets()
        self._set_default_values()

    # ── 界面构建 ────────────────────────────────────────────────

    def _create_widgets(self):
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(
            left_frame,
            text="光学规格输入 (包含物理约束)",
            font=('Arial', 12, 'bold')
        ).pack(pady=5, anchor='w')

        self._build_param_entries(left_frame)
        self._build_interactive_panel(left_frame)  
        self._build_action_buttons(left_frame)
        self._build_log_panel(left_frame)
        self._build_plot_canvas()

    def _build_param_entries(self, parent: ttk.Frame):
        entries = (
            ("广角焦距 f_wide (mm)", "f_wide"),
            ("远摄焦距 f_tele (mm)", "f_tele"),
            ("总长 TTL (mm)",        "ttl_target"),
            ("后焦距 BFD (mm)",      "bfd_target"),
            ("前组焦距 f1 (mm)",     "f1"),
            ("后组焦距 f4 (mm)",     "f4"),
            ("像面直径 Sensor (mm)", "sensor_size"),
            ("广角 F 数 F/#",        "f_number"),
            ("长焦 F 数 (T)",        "f_number_tele"),
            ("G1最大限宽 CA1 (mm)",  "max_ca1"),
            ("采样点数 N",           "num_pos"),
            ("G1 组厚度估算 (mm)",   "g1_thickness"),
            ("G2 组厚度估算 (mm)",   "g2_thickness"),
            ("G3 组厚度估算 (mm)",   "g3_thickness"),
            ("G4 组厚度估算 (mm)",   "g4_thickness"),
            # 显式等效折射率（主用字段）
            ("G1 等效折射率 n_eff₁", "n_eff_G1"),
            ("G2 等效折射率 n_eff₂", "n_eff_G2"),
            ("G3 等效折射率 n_eff₃", "n_eff_G3"),
            ("G4 等效折射率 n_eff₄", "n_eff_G4"),
            # 新增：G4 后主面偏移
            # 新增：BFL 工程下限（仅供 gauss_to_lens 诊断使用，不影响本工具求解）
            # 新增：等效阿贝数
            ("G1 等效阿贝数 V₁",     "v_eff_G1"),
            ("G2 等效阿贝数 V₂",     "v_eff_G2"),
            ("G3 等效阿贝数 V₃",     "v_eff_G3"),
            ("G4 等效阿贝数 V₄",     "v_eff_G4"),
        )

        param_frame = ttk.Frame(parent)
        param_frame.pack(fill='x', pady=5)

        half = len(entries) // 2  # 左列行数（前一半）

        for idx, (label_text, key) in enumerate(entries):
            if idx < half:
                # 左列：col 0=Label, col 1=Entry
                col_lbl, col_ent, row = 0, 1, idx
            else:
                # 右列：col 2=Label, col 3=Entry
                col_lbl, col_ent, row = 2, 3, idx - half

            ttk.Label(param_frame, text=label_text).grid(
                row=row, column=col_lbl, sticky='w', pady=2, padx=(0, 2)
            )
            entry = ttk.Entry(param_frame, width=10)
            entry.grid(row=row, column=col_ent, padx=(0, 20) if idx < half else (0, 4))
            self.params[key] = entry

        # ── "保持恒定光圈"复选框 ────────────────────────────────────
        self.var_constant_f = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(
            param_frame,
            text="保持恒定光圈 (Constant F/#)",
            variable=self.var_constant_f
        )
        chk.grid(row=half, column=0, columnspan=4, sticky='w', padx=5, pady=5)

        # ── 按钮区域 ────────────────────────────────────────────────
        btn_box = ttk.Frame(param_frame)
        btn_box.grid(row=half + 1, column=0, columnspan=4, pady=8, sticky='ew')


        ttk.Button(
            btn_box,
            text="✨ 自动估算 f1/f4",
            command=self._auto_estimate_f1_f4
        ).pack(side='left', expand=True, fill='x', padx=(0, 2))

        ttk.Button(
            btn_box,
            text="📤 导出曲线数据",
            command=self._export_curve_data
        ).pack(side='left', expand=True, fill='x', padx=(2, 0))

    def _build_interactive_panel(self, parent: ttk.Frame):
        interact_frame = ttk.LabelFrame(parent, text="🎛️ 实时交互分析 (优化后调整)", padding=5)
        interact_frame.pack(fill='x', pady=5)

        # ── 渐晕系数 ──────────────────────────────────────────────
        ttk.Label(interact_frame, text="边缘渐晕系数:").grid(row=0, column=0, sticky='w', pady=5)
        self.var_vig = tk.DoubleVar(value=0.5)
        self.spin_vig = ttk.Spinbox(
            interact_frame, from_=0.1, to=1.0, increment=0.1,
            textvariable=self.var_vig, width=8, format="%.2f",
            command=self._on_slider_drag
        )
        self.spin_vig.grid(row=0, column=1, sticky='w', padx=5)
        self.spin_vig.bind('<Return>', self._on_slider_drag)
        self.spin_vig.bind('<FocusOut>', self._on_slider_drag)

        # ── 光阑偏移 ──────────────────────────────────────────────
        ttk.Label(interact_frame, text="光阑偏移 (mm):").grid(row=1, column=0, sticky='w', pady=5)
        self.var_shift = tk.DoubleVar(value=0.0)
        self.lbl_shift = ttk.Label(interact_frame, text="0.0", width=4)
        self.lbl_shift.grid(row=1, column=2, padx=5)
        self.scale_shift = ttk.Scale(
            interact_frame, from_=-30.0, to=30.0, variable=self.var_shift,
            command=self._on_slider_drag
        )
        self.scale_shift.grid(row=1, column=1, sticky='ew', padx=5)

        # ── 光阑所在组元 ──────────────────────────────────────────
        ttk.Label(interact_frame, text="光阑所在组元:").grid(row=2, column=0, sticky='w', pady=5)
        self.var_stop_group = tk.IntVar(value=3)
        sg_frame = ttk.Frame(interact_frame)
        sg_frame.grid(row=2, column=1, columnspan=2, sticky='w', padx=5)
        for g in range(1, 5):
            ttk.Radiobutton(
                sg_frame, text=f" G{g} ", variable=self.var_stop_group, value=g,
                command=self._on_slider_drag
            ).pack(side='left', padx=2)

        interact_frame.columnconfigure(1, weight=1)

    def _build_action_buttons(self, parent: ttk.Frame):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=10)

        self.btn_run = ttk.Button(
            btn_frame,
            text="▶ 单次运行",
            command=lambda: self._start_optimization(modes=1)
        )
        self.btn_run.pack(side='left', padx=(2, 10))

        loop_frame = ttk.LabelFrame(btn_frame, text="多轮迭代设置", padding=(5, 0))
        loop_frame.pack(side='left', padx=5)

        ttk.Label(loop_frame, text="次数:").pack(side='left', padx=2)
        
        self.spin_iter = ttk.Spinbox(loop_frame, from_=1, to=100, width=3)
        self.spin_iter.set(10)  
        self.spin_iter.pack(side='left', padx=2)

        self.btn_loop = ttk.Button(
            loop_frame,
            text="🚀 循环寻优",
            command=self._run_custom_loop  
        )
        self.btn_loop.pack(side='left', padx=2)

        self.btn_exp = ttk.Button(
            btn_frame,
            text="📋 复制参数",
            command=self._copy_input_params,
            state='normal'
        )
        self.btn_exp.pack(side='left', padx=(10, 2))

        self.btn_save_fig = ttk.Button(
            btn_frame,
            text="🖼 保存图表",
            command=self._save_chart,
            state='disabled'
        )
        self.btn_save_fig.pack(side='left', padx=2)

        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)

    def _build_log_panel(self, parent: ttk.Frame):
        ttk.Label(
            parent, text="运行日志", font=('Arial', 10, 'bold')
        ).pack(anchor='w')
        self.log_text = tk.Text(
            parent, height=13, width=42, font=('Consolas', 9)
        )
        scroll = ttk.Scrollbar(parent, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(fill='both', expand=True, side='left')
        scroll.pack(fill='y', side='right')

    def _build_plot_canvas(self):
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.fig = plt.Figure(figsize=(10, 8))
        self.fig.suptitle('四组元变焦系统运动学与物理口径分析', fontsize=13)
        
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    # ── 默认值与参数处理 ────────────────────────────────────────

    def _set_default_values(self):
        defaults = dict(
            f_wide="12.0", f_tele="140.0", ttl_target="100.0",
            bfd_target="10.0", f1="80.0", f4="25.0",
            sensor_size="7.6", f_number="5.0",
            f_number_tele="5.0", max_ca1="40.0",
            num_pos="61",
        )
        # 新代码
        constant_f_state  = True
        vig_default       = 0.5
        shift_default     = 0.0
        stop_group_default = 3
        g1_t_default      = 15.0
        g2_t_default      = 9.0
        g3_t_default      = 7.0
        g4_t_default      = 21.0
        n_eff_G1_default  = 1.6
        n_eff_G2_default  = 1.8
        n_eff_G3_default  = 1.7
        n_eff_G4_default  = 1.7
        v_eff_G1_default  = 60.0
        v_eff_G2_default  = 30.0
        v_eff_G3_default  = 50.0
        v_eff_G4_default  = 55.0
        config_file = "last_run_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    for k in defaults.keys():
                        if k in saved_data:
                            defaults[k] = str(saved_data[k])
                    if "constant_f" in saved_data:
                        constant_f_state = bool(saved_data["constant_f"])
                    if "vignetting" in saved_data:
                        vig_default = float(saved_data["vignetting"])
                    if "stop_shift" in saved_data:
                        shift_default = float(saved_data["stop_shift"])
                    if "stop_group" in saved_data:
                        stop_group_default = int(saved_data["stop_group"])
                    if "n_eff_G1" in saved_data:
                        n_eff_G1_default = float(saved_data["n_eff_G1"])
                    if "n_eff_G2" in saved_data:
                        n_eff_G2_default = float(saved_data["n_eff_G2"])
                    if "n_eff_G3" in saved_data:
                        n_eff_G3_default = float(saved_data["n_eff_G3"])
                    if "n_eff_G4" in saved_data:
                        n_eff_G4_default = float(saved_data["n_eff_G4"])
                    if "v_eff_G1" in saved_data:
                        v_eff_G1_default = float(saved_data["v_eff_G1"])
                    if "v_eff_G2" in saved_data:
                        v_eff_G2_default = float(saved_data["v_eff_G2"])
                    if "v_eff_G3" in saved_data:
                        v_eff_G3_default = float(saved_data["v_eff_G3"])
                    if "v_eff_G4" in saved_data:
                        v_eff_G4_default = float(saved_data["v_eff_G4"])
            except Exception as e:
                print(f"读取上次配置失败: {e}")

        for k, v in defaults.items():
            if k in self.params:
                self.params[k].delete(0, tk.END)
                self.params[k].insert(0, v)

        self.params['g1_thickness'].insert(0, str(g1_t_default))
        self.params['g2_thickness'].insert(0, str(g2_t_default))
        self.params['g3_thickness'].insert(0, str(g3_t_default))
        self.params['g4_thickness'].insert(0, str(g4_t_default))
        # n_eff_G* 默认值（主用折射率字段）
        self.params['n_eff_G1'].insert(0, str(n_eff_G1_default))
        self.params['n_eff_G2'].insert(0, str(n_eff_G2_default))
        self.params['n_eff_G3'].insert(0, str(n_eff_G3_default))
        self.params['n_eff_G4'].insert(0, str(n_eff_G4_default))
        # 新增字段默认值
        self.params['v_eff_G1'].insert(0, str(v_eff_G1_default))
        self.params['v_eff_G2'].insert(0, str(v_eff_G2_default))
        self.params['v_eff_G3'].insert(0, str(v_eff_G3_default))
        self.params['v_eff_G4'].insert(0, str(v_eff_G4_default))

        self.var_constant_f.set(constant_f_state)
        self.var_vig.set(vig_default)
        self.var_shift.set(shift_default)
        self.var_stop_group.set(stop_group_default)
        self.lbl_shift.config(text=f"{shift_default:.1f}")

    def _export_curve_data(self):
        """导出 5 个关键位置的间距、焦距及各组元边缘光线追迹数据。

        新增三组列（共 12 列）：
            h_G1~G4     — 各组元主面处的边缘光线高度 (mm)
            u_in_G1~G4  — 各组元折射前的边缘光线角度 (rad)，G1 恒为 0（平行入射）
            u_out_G1~G4 — 各组元折射后的边缘光线角度 (rad)

        这些数据用于后续计算每个组元在各变焦位置的共轭因子 p 和形状因子 q，
        是从 Gaussian 求解到实体镜组初始结构的关键中间量。
        """
        if not self.optimizer or self.optimizer.best_trajectory is None:
            messagebox.showwarning("提示", "请先运行单次运行或循环寻优，生成轨迹数据后再导出！")
            return

        traj = self.optimizer.best_trajectory
        N = len(traj['efl'])

        # 检查新字段是否存在（兼容旧版 simulator 未更新的情况）
        has_ray_data = ('h_m' in traj and 'u_in_m' in traj and 'u_out_m' in traj)
        has_phys_ttl = 'phys_ttl' in traj

        # 计算 5 个位置的数组索引
        idx_wide = 0
        idx_mw   = N // 4
        idx_mid  = N // 2
        idx_mt   = 3 * N // 4
        idx_tele = N - 1

        indices = [
            ("短焦 (Wide)",         idx_wide),
            ("中短焦 (Medium-Wide)", idx_mw),
            ("中焦 (Medium)",        idx_mid),
            ("中长焦 (Medium-Tele)", idx_mt),
            ("长焦 (Tele)",          idx_tele),
        ]

        # ── 表头 ────────────────────────────────────────────────────────
        headers = [
            "位置",
            "焦距 EFL (mm)",
            "d1 (G1-G2间距) (mm)",
            "d2 (G2-G3间距) (mm)",
            "d3 (G3-G4间距) (mm)",
            "d1_主面间距 (Paraxial验证) (mm)",
            "d2_主面间距 (Paraxial验证) (mm)",
            "d3_主面间距 (Paraxial验证) (mm)",
            "物理 TTL (mm)",
        ]
        if has_ray_data:
            # 边缘光线高度：各组元主面处
            headers += ["h_G1 (mm)", "h_G2 (mm)", "h_G3 (mm)", "h_G4 (mm)"]
            # 入射角：折射前；G1 始终为 0（平行光入射）
            headers += ["u_in_G1 (rad)", "u_in_G2 (rad)", "u_in_G3 (rad)", "u_in_G4 (rad)"]
            # 出射角：折射后
            headers += ["u_out_G1 (rad)", "u_out_G2 (rad)", "u_out_G3 (rad)", "u_out_G4 (rad)"]

        # ── 逐位置构建行数据 ─────────────────────────────────────────────
        export_data = []
        for name, i in indices:
            row = {
                "位置":                  name,
                "焦距 EFL (mm)":         f"{traj['efl'][i]:.3f}",
                "d1 (G1-G2间距) (mm)":   f"{traj['d1'][i]:.3f}",
                "d2 (G2-G3间距) (mm)":   f"{traj['d2'][i]:.3f}",
                "d3 (G3-G4间距) (mm)":   f"{traj['d3'][i]:.3f}",
                "d1_主面间距 (Paraxial验证) (mm)": f"{traj['d1_thin'][i]:.3f}",
                "d2_主面间距 (Paraxial验证) (mm)": f"{traj['d2_thin'][i]:.3f}",
                "d3_主面间距 (Paraxial验证) (mm)": f"{traj['d3_thin'][i]:.3f}",
                "物理 TTL (mm)":         f"{traj['phys_ttl'][i]:.2f}" if has_phys_ttl else "",
            }
            if has_ray_data:
                # traj['h_m'] 形状为 (N, 4)，列顺序 G1~G4
                for g_idx, g_name in enumerate(["G1", "G2", "G3", "G4"]):
                    row[f"h_{g_name} (mm)"]      = f"{traj['h_m'][i, g_idx]:.5f}"
                    row[f"u_in_{g_name} (rad)"]  = f"{traj['u_in_m'][i, g_idx]:.6f}"
                    row[f"u_out_{g_name} (rad)"] = f"{traj['u_out_m'][i, g_idx]:.6f}"
            export_data.append(row)

        # ── 文件保存 ─────────────────────────────────────────────────────
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
            title="导出曲线数据"
        )
        if not filename:
            return

        try:
            # utf-8-sig 编码确保 Excel 打开时中文不乱码
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                # 写入元数据行（# 开头），供 Action_a 自动读取光阑位置等参数
                f.write(f"# STOP_GROUP={self.var_stop_group.get()}\n")
                f.write(f"# STOP_SHIFT={float(self.var_shift.get()):.2f}\n")
                try:
                    f.write(f"# F_NUMBER_WIDE={self.params['f_number'].get()}\n")
                    f.write(f"# F_NUMBER_TELE={self.params['f_number_tele'].get()}\n")
                    f.write(f"# SENSOR_SIZE={self.params['sensor_size'].get()}\n")
                except (KeyError, AttributeError):
                    pass  # 参数不存在时跳过，不影响主数据

                # ── 供 gauss_to_lens 端使用的 BFD_TARGET ────
                # BFD_TARGET   : 写入 Zemax LDE 的末段空气厚度（可负）
                try:
                    f.write(f"# BFL_Ideal={self.params['bfd_target'].get()}\n")
                except (KeyError, AttributeError):
                    pass  # 参数不存在时跳过，不影响主数据

                # ── 各组元焦距与通光口径（供 Action_a 自动读取填入 GUI）──
                # 焦距：traj 中 f1/f2/f3/f4 为标量常量（不随变焦位置变化）
                # 口径：traj 中 CA1~CA4 为 numpy 数组，取 max 作为组通光口径
                try:
                    f.write(f"# F_G1={traj['f1']:.3f}\n")
                    f.write(f"# F_G2={traj['f2']:.3f}\n")
                    f.write(f"# F_G3={traj['f3']:.3f}\n")
                    f.write(f"# F_G4={traj['f4']:.3f}\n")
                    f.write(f"# D_G1={float(max(traj['CA1'])):.2f}\n")
                    f.write(f"# D_G2={float(max(traj['CA2'])):.2f}\n")
                    f.write(f"# D_G3={float(max(traj['CA3'])):.2f}\n")
                    f.write(f"# D_G4={float(max(traj['CA4'])):.2f}\n")
                except (KeyError, TypeError):
                    pass  # traj 缺字段时跳过，不影响主数据

                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(export_data)

            # 日志预览（保持原有风格，新增光线数据摘要）
            self._log(f"\n>>> 成功导出 5 个关键位置的曲线数据至:\n    {filename}")
            for row in export_data:
                line = (
                    f"    [{row['位置']}] "
                    f"EFL: {row['焦距 EFL (mm)']}, "
                    f"d1: {row['d1 (G1-G2间距) (mm)']}, "
                    f"d2: {row['d2 (G2-G3间距) (mm)']}, "
                    f"d3: {row['d3 (G3-G4间距) (mm)']}"
                )
                if has_ray_data:
                    line += (
                        f" | h(G1-G4): "
                        f"{row['h_G1 (mm)']}/"
                        f"{row['h_G2 (mm)']}/"
                        f"{row['h_G3 (mm)']}/"
                        f"{row['h_G4 (mm)']} mm"
                    )
                self._log(line)

            messagebox.showinfo("成功", f"数据已成功导出至:\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")

    # ── 实时交互逻辑 (滑块/微调框防抖) ──────────────────────────────────

    def _on_slider_drag(self, event=None):
        try:
            # 捕获可能的非法输入(如用户在spinbox中输入了字母)
            vig_val = self.var_vig.get()
        except tk.TclError:
            return

        shift_val = self.var_shift.get()
        # 【修改点 4】：移除 lbl_vig 的更新
        self.lbl_shift.config(text=f"{shift_val:.1f}")

        if not self.optimizer or self.optimizer.best_params is None:
            return

        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        
        self._after_id = self.root.after(50, self._do_interactive_replot)

    # 新代码
    def _do_interactive_replot(self):
        try:
            self.optimizer.config.vignetting = self.var_vig.get()
        except tk.TclError:
            pass  # 忽略无效输入

        self.optimizer.config.stop_shift = self.var_shift.get()
        self.optimizer.config.stop_group = self.var_stop_group.get()

        f2, f3, m2_W, m2_T, f1_fac, f4_fac = self.optimizer.best_params
        f1_dyn = self.optimizer.config.f1 * f1_fac
        f4_dyn = self.optimizer.config.f4 * f4_fac
        traj = self.optimizer.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        if traj is not None:
            self.optimizer.best_trajectory = traj
            self._update_plots()

    # ── 优化调度 ────────────────────────────────────────────────

    def _auto_estimate_f1_f4(self):
        try:
            ttl = float(self.params['ttl_target'].get())
            bfd = float(self.params['bfd_target'].get())
            f1_est = 0.8 * ttl
            f4_est = 2.5 * bfd

            self.params['f1'].config(state='normal')
            self.params['f1'].delete(0, tk.END)
            self.params['f1'].insert(0, f"{f1_est:.1f}")
            self.params['f1'].config(state='readonly')

            self.params['f4'].delete(0, tk.END)
            self.params['f4'].insert(0, f"{f4_est:.1f}")

            self._log(f" 已应用光学经验公式：f1≈{f1_est:.1f}, f4={f4_est:.1f}")
        except ValueError:
            messagebox.showwarning("提示", "请先输入有效的 TTL 和 BFD！")

    def _log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _run_custom_loop(self):
        """读取用户输入的迭代次数并启动优化"""
        try:
            val = self.spin_iter.get()
            n_loops = int(val)
            
            if n_loops < 1:
                messagebox.showwarning("提示", "迭代次数至少为 1 次！")
                return
            
            self._start_optimization(modes=n_loops)
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数次数！")
    
    def _start_optimization(self, modes=1):
        if not self.params['f1'].get().strip() or not self.params['f4'].get().strip():
            self._auto_estimate_f1_f4()

        # 新代码
        save_data = {k: entry.get() for k, entry in self.params.items()}
        save_data["constant_f"]   = self.var_constant_f.get()
        save_data["vignetting"]   = self.var_vig.get()
        save_data["stop_shift"]   = self.var_shift.get()
        save_data["stop_group"]   = self.var_stop_group.get()

        try:
            with open("last_run_config.json", 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4)
        except Exception as e:
            print(f"保存配置失败: {e}")

        try:
            # 新代码
            cfg = ZoomConfig(
                f_wide            = float(self.params['f_wide'].get()),
                f_tele            = float(self.params['f_tele'].get()),
                ttl_target        = float(self.params['ttl_target'].get()),
                bfd_target        = float(self.params['bfd_target'].get()),
                f1                = float(self.params['f1'].get()),
                f4                = float(self.params['f4'].get()),
                sensor_size       = float(self.params['sensor_size'].get()),
                f_number          = float(self.params['f_number'].get()),
                f_number_tele     = float(self.params['f_number_tele'].get()),
                max_ca1           = float(self.params['max_ca1'].get()),
                num_positions     = int(self.params['num_pos'].get()),
                stop_shift        = self.var_shift.get(),
                stop_group        = self.var_stop_group.get(),
                vignetting        = self.var_vig.get(),
                constant_f_number = self.var_constant_f.get(),
                g1_thickness      = float(self.params['g1_thickness'].get()),
                g2_thickness      = float(self.params['g2_thickness'].get()),
                g3_thickness      = float(self.params['g3_thickness'].get()),
                g4_thickness      = float(self.params['g4_thickness'].get()),
                n_eff_G1          = float(self.params['n_eff_G1'].get()),
                n_eff_G2          = float(self.params['n_eff_G2'].get()),
                n_eff_G3          = float(self.params['n_eff_G3'].get()),
                n_eff_G4          = float(self.params['n_eff_G4'].get()),
                v_eff_G1          = float(self.params['v_eff_G1'].get()),
                v_eff_G2          = float(self.params['v_eff_G2'].get()),
                v_eff_G3          = float(self.params['v_eff_G3'].get()),
                v_eff_G4          = float(self.params['v_eff_G4'].get()),
            )
        except ValueError:
            messagebox.showerror("错误", "参数格式不正确，请检查所有输入框！")
            return

        self.log_text.delete(1.0, tk.END)
        self.btn_run.config(state='disabled')
        self.btn_loop.config(state='disabled')
        self.btn_exp.config(state='disabled')
        self.btn_save_fig.config(state='disabled')
        self.progress.start()

        threading.Thread(
            target=self._optimize_worker, args=(cfg, modes), daemon=True
        ).start()

    def _optimize_worker(self, config: ZoomConfig, n_runs: int):
        try:
            _t_start = time.perf_counter()
            self.root.after(0, self._log, f">>> 启动工程级光学寻优 (共 {n_runs} 轮)")
            self.root.after(0, self._log, f"    目标: {config.f_wide}-{config.f_tele}mm | TTL: {config.ttl_target}mm")

            best_optimizer = None
            best_score = float('inf')

            for i in range(1, n_runs + 1):
                prefix = f"[第{i}/{n_runs}轮]"
                if n_runs > 1:
                    self.root.after(0, self._log, f"\n--- {prefix} 正在计算... ---")

                current_optimizer = ZoomLensOptimizer(config)
                # 第1轮用默认种子，后续轮次使用额外种子探索不同区域
                extra_seeds = [i * 100 + j for j in range(1, 4)] if i > 1 else None
                success = current_optimizer.optimize(callback=None, extra_seeds=extra_seeds)

                if current_optimizer.best_params is not None:
                    breakdown = current_optimizer.get_penalty_diagnostics(current_optimizer.best_params)
                    current_loss = breakdown.get("总分 (Total)", float('inf'))
                    
                    msg = f"    ✓ 得分: {current_loss:.2e}"
                    if current_loss < best_score:
                        best_score = current_loss
                        best_optimizer = current_optimizer
                        msg += " (👑 新纪录!)"
                    
                    if n_runs > 1:
                        self.root.after(0, self._log, msg)
                else:
                    if n_runs > 1:
                        self.root.after(0, self._log, f"    ✗ 未收敛")

            if best_optimizer is not None:
                self.optimizer = best_optimizer
                self.root.after(0, self._log, f"\n🏆 最终结果 (最低得分: {best_score:.2e})")

                breakdown = self.optimizer.get_penalty_diagnostics(self.optimizer.best_params)
                msg = ["📊 【优化结果深度诊断报告】"]
                total_loss = breakdown.get("总分 (Total)", 1e-9)
                
                sorted_items = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
                for key, val in sorted_items:
                    if key == "总分 (Total)": continue
                    pct = (val / total_loss) * 100 if total_loss > 0 else 0
                    if val > 1.0:
                        tag = "🔴" if pct > 10 else "  " 
                        msg.append(f"  {tag} {key}: {val:.2e} (占比 {pct:.1f}%)")
                self.root.after(0, self._log, "\n".join(msg))

                advice_msg =["\n💡 【系统智能调参建议】"]
                has_advice = False
                ca1_pct = breakdown.get("8. G1口径超标 (CA1)", 0) / total_loss
                efl_pct = breakdown.get("2. 焦距误差 (EFL)", 0) / total_loss
                
                if ca1_pct > 0.15:
                    has_advice = True
                    advice_msg.append("👉 [前组口径爆炸] 药方：尝试在左侧大幅增大初始 f4，或调整边缘渐晕系数。")
                if efl_pct > 0.15:
                    has_advice = True
                    advice_msg.append("👉 [长焦/广角未达标] 药方：1. 增大总长 TTL； 2. 尝试将 f1 设为接近目标长焦焦距。")

                if not has_advice:
                    advice_msg.append("👉 系统状态良好！可调整下方参数实时分析光阑偏移与渐晕对口径的影响。")

                self.root.after(0, self._log, "\n".join(advice_msg))
                self.root.after(0, self._on_success)
            else:
                self.root.after(
                    0, lambda: messagebox.showwarning("优化失败", "未能找到有效解。")
                )

        except Exception:
            import traceback
            err_msg = traceback.format_exc()
            self.root.after(0, lambda: self._log(f"\n{err_msg}"))
        finally:
            print(f"[TIMING] auto 总耗时: {time.perf_counter() - _t_start:.1f}s ({n_runs} 轮)")
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.btn_run.config(state='normal'))
            self.root.after(0, lambda: self.btn_loop.config(state='normal'))

    def _on_success(self):
        self.btn_exp.config(state='normal')
        self.btn_save_fig.config(state='normal')
        traj = self.optimizer.best_trajectory
        
        f2, f3, m2_W, m2_T, f1_fac, f4_fac = self.optimizer.best_params
        sys_obj = self.optimizer.system

        best_f1 = sys_obj.config.f1 * f1_fac
        best_f4 = sys_obj.config.f4 * f4_fac

        self._log("\n>>> 优化成功！核心参数分配:")
        self._log(f"    f1 = {best_f1:.3f} mm  (初始值: {sys_obj.config.f1:.1f}, 浮动: {f1_fac:.2f}x)")
        self._log(f"    f2 = {f2:.3f} mm")
        self._log(f"    f3 = {f3:.3f} mm")
        self._log(f"    f4 = {best_f4:.3f} mm  (初始值: {sys_obj.config.f4:.1f}, 浮动: {f4_fac:.2f}x)")

        cfg = self.optimizer.config
        ttl_actual = self.optimizer.best_ttl
        self._log(f"  优化器选 TTL (z_G4_ref+BFD) = {ttl_actual:.2f} mm (目标 {cfg.ttl_target}, 偏差 {(ttl_actual/cfg.ttl_target - 1)*100:+.1f}%)")
        # 物理 TTL = 薄透镜 TTL + (t_G1 + t_G4)/2，因为 d1/d2/d3 已做半厚度修正
        phys_ttl_val = ttl_actual + (sys_obj._t_G1 + sys_obj._t_G4) / 2.0
        self._log(f"  物理 TTL (含组厚度) = {phys_ttl_val:.2f} mm (相对目标偏差 {(phys_ttl_val/cfg.ttl_target - 1)*100:+.1f}%)")
        self._log(f"  半厚度补偿 = {(sys_obj._t_G1 + sys_obj._t_G4)/2.0:.2f} mm (= (t_G1 + t_G4) / 2)")
        ttl_max_val = cfg.ttl_target * 1.5
        self._log(f"  TTL 硬上限 (1.5 × 目标) = {ttl_max_val:.2f} mm")
        if phys_ttl_val > ttl_max_val:
            self._log(f"  ⚠️ 物理 TTL 超过硬上限 {phys_ttl_val - ttl_max_val:.2f} mm")

        P_sum = (
            (1.0 / best_f1) / cfg.n_eff_G1 +
            (1.0 / f2)     / cfg.n_eff_G2 +
            (1.0 / f3)     / cfg.n_eff_G3 +
            (1.0 / best_f4) / cfg.n_eff_G4
        )
        R_pz = -1.0 / P_sum if P_sum != 0 else float('inf')
        self._log(f"    *系统佩兹伐和 (分组 n) = {P_sum:.4f}")
        self._log(f"    *初始场曲曲率半径 R_pz = {R_pz:.1f} mm")
        
        crossings = int(np.sum(np.diff(np.sign(traj['m3'] - (-1.0))) != 0))
        if crossings > 0:
            pct = np.argmin(np.abs(traj['m3'] - (-1.0))) / (sys_obj.config.num_positions - 1) * 100
            self._log(f"    *换根检测: ✓ 成功物理换根 (发生于行程 {pct:.1f}% 处)")
        else:
            self._log(f"    *换根检测: ✗ 未发生换根 (单根运行)")
            
        
        sg = self.optimizer.config.stop_group
        ca_names = {1: ['G1(STOP)', 'G2', 'G3', 'G4'],
                    2: ['G1', 'G2(STOP)', 'G3', 'G4'],
                    3: ['G1', 'G2', 'G3(STOP)', 'G4'],
                    4: ['G1', 'G2', 'G3', 'G4(STOP)']}
        lbl = ca_names.get(sg, ca_names[3])
        self._log(f"\n>>> 各组最大通光孔径估计(CA):")
        self._log(f"    {lbl[0]}: {np.max(traj['CA1']):.1f}mm | {lbl[1]}: {np.max(traj['CA2']):.1f}mm")
        self._log(f"    {lbl[2]}: {np.max(traj['CA3']):.1f}mm | {lbl[3]}: {np.max(traj['CA4']):.1f}mm")

        self._update_plots()

    def _update_plots(self):
        print("DEBUG: _update_plots called")
        try:
            if not self.optimizer:
                return
            traj = self.optimizer.best_trajectory
            cfg = self.optimizer.config
            sys_obj = self.optimizer.system
            x = np.linspace(0, 1, cfg.num_positions)

            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()

            self.ax1.plot(x, traj['root1_z3'], '.', color='#CCCCCC', markersize=2)
            self.ax1.plot(x, traj['root2_z3'], '.', color='#AAAAAA', markersize=2)
            self.ax1.plot(x, traj['z3'], 'r-', linewidth=2, label='G3 (补偿)')
            self.ax1.plot(x, traj['z2'], 'b--', linewidth=2, label='G2 (变倍)')
            self.ax1.axhline(sys_obj.z_G1, color='k', linestyle=':', label='G1')
            self.ax1.axhline(sys_obj.z_G4_ref, color='g', linestyle=':', label='G4')
            self.ax1.set_title('凸轮曲线运动轨迹')
            self.ax1.set_ylabel('透镜位置 z (mm)')
            self.ax1.legend(loc='best', fontsize=8)
            self.ax1.grid(True, alpha=0.3)

            self.ax2.plot(x, traj['efl'], 'g-', linewidth=2)
            self.ax2.axhline(cfg.f_wide, color='r', linestyle='--')
            self.ax2.axhline(cfg.f_tele, color='b', linestyle='--')
            self.ax2.set_yscale('log')
            self.ax2.set_title('系统焦距 EFL 验证')
            self.ax2.grid(True, which='both', alpha=0.3)

            self.ax3.plot(x, traj['m3'], 'm-', linewidth=2)
            self.ax3.axhline(-1.0, color='r', linestyle='--')
            self.ax3.set_title('G3 放大率 m3')
            self.ax3.grid(True, alpha=0.3)

            # 新代码
            sg = cfg.stop_group if hasattr(cfg, 'stop_group') else 3
            _base = ['G1', 'G2', 'G3', 'G4']
            _base[sg - 1] = f'G{sg}(STOP)'
            self.ax4.plot(x, traj['CA1'], 'k-', label=_base[0])
            self.ax4.plot(x, traj['CA2'], 'b-', label=_base[1])
            self.ax4.plot(x, traj['CA3'], 'r-', label=_base[2])
            self.ax4.plot(x, traj['CA4'], 'g-', label=_base[3])

            max_ca1_val = np.max(traj['CA1'])
            self.ax4.set_title(f'各组通光孔径需求 (G1 Max: {max_ca1_val:.1f}mm)')
            self.ax4.set_ylabel('直径 CA (mm)')
            self.ax4.legend(loc='best', fontsize=8)
            self.ax4.grid(True, alpha=0.3)

            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            import traceback; traceback.print_exc()
        print("DEBUG: _update_plots done")

    def _copy_input_params(self):
        try:
            lines =[
                "【光学变焦镜头设计 - 当前输入参数】",
                "-" * 40,
                f"广角焦距 f_wide    : {self.params['f_wide'].get()} mm",
                f"远摄焦距 f_tele    : {self.params['f_tele'].get()} mm",
                f"总长 TTL           : {self.params['ttl_target'].get()} mm",
                f"后焦距 BFD         : {self.params['bfd_target'].get()} mm",
                f"前组焦距 f1        : {self.params['f1'].get()} mm",
                f"后组焦距 f4        : {self.params['f4'].get()} mm",
                f"像面直径 Sensor    : {self.params['sensor_size'].get()} mm",
                f"广角 F 数 F/#      : {self.params['f_number'].get()}",
                f"长焦 F 数 (T)      : {self.params['f_number_tele'].get()}",
                f"G1最大限宽 CA1     : {self.params['max_ca1'].get()} mm",
                f"采样点数 N         : {self.params['num_pos'].get()}",
                "-" * 40,
                f"保持恒定光圈       : {'是' if self.var_constant_f.get() else '否'}",
                f"边缘渐晕系数       : {self.var_vig.get():.2f}",
                f"光阑偏移           : {self.var_shift.get():.1f} mm",
                "-" * 40
            ]
            
            copy_text = "\n".join(lines)
            self.root.clipboard_clear()
            self.root.clipboard_append(copy_text)
            self.root.update()  
            
            messagebox.showinfo("成功", "所有输入参数已成功复制到剪贴板！\n您可以直接粘贴发送。")
        except Exception as e:
            messagebox.showerror("错误", f"复制参数失败: {str(e)}")

    def _save_chart(self):
        if not self.optimizer: return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG 高清图像", "*.png"), ("JPEG 图像", "*.jpg"), ("所有文件", "*.*")),
            title="保存分析图表"
        )
        if not filename: return
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"图表已成功保存至:\n{filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图片失败: {str(e)}")