import numpy as np
from scipy.optimize import differential_evolution, minimize
from config import ZoomConfig
from simulator import ZoomSystemSimulator

import os as _os

_DE_WORKERS = min(int(_os.environ.get('GAUSSIAN_DE_WORKERS', '0')) or (_os.cpu_count() or 2), _os.cpu_count() or 2)

class ZoomLensOptimizer:
    def __init__(self, config: ZoomConfig):
        self.config = config
        self.system = ZoomSystemSimulator(config)
        self.best_params = None
        self.best_trajectory = None
        self._current_ttl = config.ttl_target  # objective 内重算 z_G4_ref 用

        ref_scale = 10.0
        current_scale = self.config.f_wide if self.config.f_wide > 0.1 else 10.0
        ratio = current_scale / ref_scale

        self.weights = {
            'efl': 5.0e5,
            'root_force': 1.0e9,
            'root_center': 1.0e4,
            'gaps': 5.0e7 / ratio,
            'monotonicity': 1.0e5 / ratio,
            'petzval': 5.0e5 * ratio,
            'smoothness': 50.0 / (ratio ** 2),
            'delta': 5.0e7 / (ratio ** 2),
            'ca1_limit': 5.0e4 / ratio,
            'chromatic': 3.0e5 * ratio,  # 组级色差软约束
            'bfd_min': 5.0e7 / ratio,  # BFD 软下限（与 gaps 同档）
        }

    def _compute_bounds(self):
        f_w = self.config.f_wide
        f_t = self.config.f_tele
        zoom_ratio = f_t / f_w

        return (
            (-f_t * 1.2, -5.0),         # f2
            (10.0, f_t * 1.5),          # f3
            (-2.0, -0.1),               # m2_W
            (-zoom_ratio * 2.0, -0.5),  # m2_T
            (20.0, max(80.0, f_t * 0.85)),  # f1 (绝对焦距 mm)
            (40.0, 80.0),               # f4 (绝对焦距 mm)
            (0.5 * self.config.bfd_min, 20.0),  # bfd（软下限 ≥ bfd_min）
        )

    def objective_function(self, params: np.ndarray) -> float:
        f2, f3, m2_W, m2_T, f1_dyn, f4_dyn, bfd = params

        if m2_W <= m2_T:
            return 1e9

        # BFD 为搜索变量：每次评估同步 override 与 G4 位置，保证 TTL 守恒
        self.system.bfd_override = bfd
        self.system.z_G4_ref = self._current_ttl - bfd

        traj = self.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        # 数学越界（G3 无实数解）→ 直接返回大 penalty，避免虚假 m3 解污染下游 EFL
        if traj['delta_violation'] > 0:
            return 1e9 + traj['delta_violation'] * 1e6

        z2, z3, efl, m3 = traj['z2'], traj['z3'], traj['efl'], traj['m3']
        d1, d2, d3 = traj['d1'], traj['d2'], traj['d3']
        CA1, CA2, CA3, CA4 = traj['CA1'], traj['CA2'], traj['CA3'], traj['CA4']

        penalty = 0.0
        penalty += traj['delta_violation'] * self.weights['delta']
        penalty += self._penalty_efl(efl)
        penalty += self._penalty_gaps(d1, d2, d3, CA1, CA2, CA3, CA4)
        penalty += self._penalty_petzval(f1_dyn, f2, f3, f4_dyn)
        penalty += self._penalty_chromatic(f1_dyn, f2, f3, f4_dyn)
        penalty += self._penalty_monotonicity(z2, z3)
        penalty += self._penalty_smoothness(z3)
        penalty += self._penalty_ca1(CA1)
        penalty += self._penalty_root_center(m3)
        penalty += self._penalty_bfd(bfd)

        return penalty

    def optimize(self, callback=None, extra_seeds=None) -> bool:
        # TTL 候选系数：覆盖 0.95×~1.5× 范围，扫描紧凑到宽松设计
        ttl_candidates = [self.config.ttl_target * r for r in [1.05, 1.1, 1.2, 1.3]]
        ttl_max = self.config.ttl_target * 1.5

        if callback:
            callback(f"扫描 {len(ttl_candidates)} 个 TTL 候选值...")

        best_fun = float('inf')
        best_x = None
        best_ttl = self.config.ttl_target

        bounds = self._compute_bounds()  # 6维

        for ttl_val in ttl_candidates:
            self._current_ttl = ttl_val  # objective 据此 + candidate bfd 重算 z_G4_ref

            best_res_fun = float('inf')
            best_res_x = None
            if extra_seeds:
                # 第2轮起：只跑新种子，跳过结果恒定的固定种子
                seeds = list(extra_seeds)
            else:
                # 第1轮：跑固定种子建立基线
                seeds = [42, 123, 7, 2024, 99]
            for s in seeds:
                res = differential_evolution(
                    self.objective_function, bounds,
                    strategy='best1bin', maxiter=250, popsize=40,
                    tol=0.005, seed=s, workers=_DE_WORKERS, disp=False
                )
                if res.fun < best_res_fun:
                    best_res_fun = res.fun
                    best_res_x = res.x.copy()

            # 基于物理 TTL（含半厚度补偿）的双层惩罚
            phys_ttl = ttl_val + (self.system._t_G1 + self.system._t_G4) / 2.0
            ttl_over = max(0.0, phys_ttl - ttl_max)
            penalty_ttl_hard = (ttl_over ** 2) * 1.0e7
            ttl_dev_norm = (phys_ttl - self.config.ttl_target) / self.config.ttl_target
            penalty_ttl_soft = (ttl_dev_norm ** 2) * 1.0e3
            total = best_res_fun + penalty_ttl_hard + penalty_ttl_soft

            if callback:
                callback(f"  TTL={ttl_val:.0f} -> {total:.2e}")

            if total < best_fun:
                best_fun = total
                best_x = best_res_x.copy()
                best_ttl = ttl_val

        # 精修
        self._current_ttl = best_ttl

        if callback:
            callback(f"最佳 TTL={best_ttl:.0f}，单纯形精修...")

        res_local = minimize(
            self.objective_function, best_x,
            method='Nelder-Mead',
            options=dict(xatol=1e-6, fatol=1e-6, maxiter=5000)
        )

        # Nelder-Mead 不尊重 bounds，对精修结果做边界裁剪
        bounds = self._compute_bounds()
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        clipped_x = np.clip(res_local.x, lower, upper)

        # 始终使用裁剪后的解（越界解不可信）
        final_x = clipped_x
        self.best_params = final_x
        self.best_ttl = best_ttl
        self.best_phys_ttl = best_ttl + (self.system._t_G1 + self.system._t_G4) / 2.0

        f2, f3, m2_W, m2_T, f1_dyn, f4_dyn, bfd = final_x
        self.best_bfd = bfd
        self.system.bfd_override = bfd
        self.system.z_G4_ref = best_ttl - bfd
        self.best_trajectory = self.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        return True

    def get_penalty_diagnostics(self, params: np.ndarray) -> dict:
        f2, f3, m2_W, m2_T, f1_dyn, f4_dyn, bfd = params

        if m2_W <= m2_T:
            return {"致命错误": "广角/长焦放大率倒置 (m2_W <= m2_T)"}

        self.system.bfd_override = bfd
        self.system.z_G4_ref = self._current_ttl - bfd

        traj = self.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        z2, z3, efl, m3 = traj['z2'], traj['z3'], traj['efl'], traj['m3']
        d1, d2, d3 = traj['d1'], traj['d2'], traj['d3']
        CA1, CA2, CA3, CA4 = traj['CA1'], traj['CA2'], traj['CA3'], traj['CA4']

        p_delta = traj['delta_violation'] * self.weights['delta']
        p_efl = self._penalty_efl(efl)
        p_gaps = self._penalty_gaps(d1, d2, d3, CA1, CA2, CA3, CA4)
        p_petzval = self._penalty_petzval(f1_dyn, f2, f3, f4_dyn)
        p_chrom = self._penalty_chromatic(f1_dyn, f2, f3, f4_dyn)
        p_mono = self._penalty_monotonicity(z2, z3)
        p_smooth = self._penalty_smoothness(z3)
        p_root = self._penalty_root_center(m3)
        p_ca1 = self._penalty_ca1(CA1)
        p_bfd = self._penalty_bfd(bfd)

        total = p_delta + p_efl + p_gaps + p_petzval + p_chrom + p_mono + p_smooth + p_root + p_ca1 + p_bfd

        return {
            "总分 (Total)": total,
            "1. 数学越界 (Delta)": p_delta,
            "2. 焦距误差 (EFL)": p_efl,
            "3. 物理防撞 (Gaps)": p_gaps,
            "4. 场曲控制 (Petzval)": p_petzval,
            "4b. 色差约束 (Chrom)": p_chrom,
            "5. G2单调性 (Mono)": p_mono,
            "6. 轨迹平滑 (Smooth)": p_smooth,
            "7. 换根居中度 (Root)": p_root,
            "8. G1口径超标 (CA1)": p_ca1,
            "9. BFD软下限 (BFD)": p_bfd
        }

    # ── 辅助函数保持不变 ──
    def _penalty_efl(self, efl: np.ndarray) -> float:
        return (
            ((efl[0]  - self.config.f_wide) / self.config.f_wide) ** 2 +
            ((efl[-1] - self.config.f_tele) / self.config.f_tele) ** 2
        ) * self.weights['efl']

    def _penalty_gaps(self, d1, d2, d3, CA1, CA2, CA3, CA4) -> float:
        min_air_gap = 2.0
        penalty = np.sum(np.maximum(0.0, min_air_gap - d1) ** 2)
        penalty += np.sum(np.maximum(0.0, min_air_gap - d2) ** 2)
        penalty += np.sum(np.maximum(0.0, min_air_gap - d3) ** 2)
        return penalty * self.weights['gaps']

    def _penalty_petzval(self, f1_dyn: float, f2: float, f3: float, f4_dyn: float) -> float:
        # 各组使用独立等效折射率，而非统一假设或统一 n_G*
        # 计算格式：(1/f_i) / n_eff_i
        P_sum = (
            (1.0 / f1_dyn) / self.config.n_eff_G1 +
            (1.0 / f2)     / self.config.n_eff_G2 +
            (1.0 / f3)     / self.config.n_eff_G3 +
            (1.0 / f4_dyn) / self.config.n_eff_G4
        )
        # Petzval 死区 0.025：原 0.015 对高倍率(≥14×)过紧，误伤 EFL 可行解；
        # 实测 14×(10/140) 放宽后 total -23%、P_sum 仅用到 -0.0149(场曲未实际变差），
        # 而 ≤10× 样本 P_sum 量级 ~0.001，远在死区内、放宽零影响。
        return max(0.0, abs(P_sum) - 0.025) * self.weights['petzval']

    def _penalty_chromatic(self, f1_dyn: float, f2: float, f3: float, f4_dyn: float) -> float:
        """
        组级初级色差软约束：Σ(φᵢ / Vᵢ) ≈ 0
        φᵢ = 1/fᵢ 为各组光焦度，Vᵢ 为各组等效阿贝数。
        该约束为软约束（惩罚项），不强制为零，允许小量残余。
        """
        chrom_sum = (
            (1.0 / f1_dyn) / self.config.v_eff_G1 +
            (1.0 / f2)     / self.config.v_eff_G2 +
            (1.0 / f3)     / self.config.v_eff_G3 +
            (1.0 / f4_dyn) / self.config.v_eff_G4
        )
        return chrom_sum ** 2 * self.weights['chromatic']

    def _penalty_monotonicity(self, z2: np.ndarray, z3: np.ndarray) -> float:
        def _mono(z):
            dz = np.diff(z)
            main_dir = np.sign(np.sum(dz)) if np.sum(dz) != 0 else 1.0
            return np.sum(np.maximum(0.0, -dz * main_dir))
        return _mono(z2) * self.weights['monotonicity']

    def _penalty_smoothness(self, z3: np.ndarray) -> float:
        return np.sum(np.diff(z3, n=2) ** 2) * self.weights['smoothness']

    def _penalty_root_center(self, m3: np.ndarray) -> float:
        min_dist = np.min(np.abs(m3 - (-1.0)))
        if np.max(m3) < -1.0 or np.min(m3) > -1.0:
            return min_dist * self.weights['root_force']
        cross_idx = np.argmin(np.abs(m3 - (-1.0)))
        target_idx = len(m3) // 2
        return (abs(cross_idx - target_idx) / len(m3)) ** 2 * self.weights['root_center']

    def _penalty_bfd(self, bfd: float) -> float:
        return max(0.0, self.config.bfd_min - bfd) ** 2 * self.weights['bfd_min']

    def _penalty_ca1(self, CA1: np.ndarray) -> float:
        violation = np.sum(np.maximum(0.0, CA1 - self.config.max_ca1))
        return violation * self.weights['ca1_limit']


def build_summary_lines(optimizer) -> list:
    lines = []

    traj = optimizer.best_trajectory

    f2, f3, m2_W, m2_T, best_f1, best_f4, bfd = optimizer.best_params
    sys_obj = optimizer.system
    cfg = optimizer.config

    lines.append("\n>>> 优化成功！核心参数分配:")
    lines.append(f"    f1 = {best_f1:.3f} mm  (自由优化)")
    lines.append(f"    f2 = {f2:.3f} mm")
    lines.append(f"    f3 = {f3:.3f} mm")
    lines.append(f"    f4 = {best_f4:.3f} mm  (自由优化)")
    _zoom_dbg = cfg.f_tele / cfg.f_wide
    lines.append(f"    m2_W = {m2_W:.3f} (边界 -2.0 ~ -0.1) | m2_T = {m2_T:.3f} (边界 {-_zoom_dbg*2.0:.2f} ~ -0.5)")
    lines.append(f"    BFD = {traj['bfd']:.3f} mm  (下限 {sys_obj.config.bfd_min:.1f})")

    ttl_actual = optimizer.best_ttl
    lines.append(f"  优化器选 TTL (z_G4_ref+BFD) = {ttl_actual:.2f} mm (目标 {cfg.ttl_target}, 偏差 {(ttl_actual/cfg.ttl_target - 1)*100:+.1f}%)")
    # 物理 TTL = 薄透镜 TTL + (t_G1 + t_G4)/2，因为 d1/d2/d3 已做半厚度修正
    phys_ttl_val = ttl_actual + (sys_obj._t_G1 + sys_obj._t_G4) / 2.0
    lines.append(f"  物理 TTL (含组厚度) = {phys_ttl_val:.2f} mm (相对目标偏差 {(phys_ttl_val/cfg.ttl_target - 1)*100:+.1f}%)")
    lines.append(f"  半厚度补偿 = {(sys_obj._t_G1 + sys_obj._t_G4)/2.0:.2f} mm (= (t_G1 + t_G4) / 2)")
    ttl_max_val = cfg.ttl_target * 1.5
    lines.append(f"  TTL 硬上限 (1.5 × 目标) = {ttl_max_val:.2f} mm")
    if phys_ttl_val > ttl_max_val:
        lines.append(f"  ⚠️ 物理 TTL 超过硬上限 {phys_ttl_val - ttl_max_val:.2f} mm")

    P_sum = (
        (1.0 / best_f1) / cfg.n_eff_G1 +
        (1.0 / f2)     / cfg.n_eff_G2 +
        (1.0 / f3)     / cfg.n_eff_G3 +
        (1.0 / best_f4) / cfg.n_eff_G4
    )
    R_pz = -1.0 / P_sum if P_sum != 0 else float('inf')
    lines.append(f"    *系统佩兹伐和 (分组 n) = {P_sum:.4f}")
    lines.append(f"    *初始场曲曲率半径 R_pz = {R_pz:.1f} mm")

    crossings = int(np.sum(np.diff(np.sign(traj['m3'] - (-1.0))) != 0))
    if crossings > 0:
        pct = np.argmin(np.abs(traj['m3'] - (-1.0))) / (sys_obj.config.num_positions - 1) * 100
        lines.append(f"    *换根检测: ✓ 成功物理换根 (发生于行程 {pct:.1f}% 处)")
    else:
        lines.append(f"    *换根检测: ✗ 未发生换根 (单根运行)")

    sg = optimizer.config.stop_group
    ca_names = {1: ['G1(STOP)', 'G2', 'G3', 'G4'],
                2: ['G1', 'G2(STOP)', 'G3', 'G4'],
                3: ['G1', 'G2', 'G3(STOP)', 'G4'],
                4: ['G1', 'G2', 'G3', 'G4(STOP)']}
    lbl = ca_names.get(sg, ca_names[3])
    lines.append(f"\n>>> 各组最大通光孔径估计(CA):")
    lines.append(f"    {lbl[0]}: {np.max(traj['CA1']):.1f}mm | {lbl[1]}: {np.max(traj['CA2']):.1f}mm")
    lines.append(f"    {lbl[2]}: {np.max(traj['CA3']):.1f}mm | {lbl[3]}: {np.max(traj['CA4']):.1f}mm")

    return lines
