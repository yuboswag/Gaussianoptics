import numpy as np
from scipy.optimize import differential_evolution, minimize
from config import ZoomConfig
from simulator import ZoomSystemSimulator

class ZoomLensOptimizer:
    def __init__(self, config: ZoomConfig):
        self.config = config
        self.system = ZoomSystemSimulator(config)
        self.best_params = None
        self.best_trajectory = None

        ref_scale = 10.0
        current_scale = self.config.f_wide if self.config.f_wide > 0.1 else 10.0
        ratio = current_scale / ref_scale

        self.weights = {
            'efl': 5.0e4,
            'root_force': 1.0e9,
            'root_center': 1.0e6,
            'gaps': 5.0e7 / ratio,
            'monotonicity': 1.0e5 / ratio,
            'petzval': 5.0e5 * ratio,
            'smoothness': 50.0 / (ratio ** 2),
            'delta': 5.0e7 / (ratio ** 2),
            'ca1_limit': 5.0e4 / ratio
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
            (0.7, 1.3),                 # f1_factor
            (0.5, 2.0),                 # f4_factor
        )

    def objective_function(self, params: np.ndarray) -> float:
        f2, f3, m2_W, m2_T, f1_fac, f4_fac = params

        if m2_W <= m2_T:
            return 1e9

        f1_dyn = self.config.f1 * f1_fac
        f4_dyn = self.config.f4 * f4_fac

        traj = self.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        z2, z3, efl, m3 = traj['z2'], traj['z3'], traj['efl'], traj['m3']
        d1, d2, d3 = traj['d1'], traj['d2'], traj['d3']
        CA1, CA2, CA3, CA4 = traj['CA1'], traj['CA2'], traj['CA3'], traj['CA4']

        penalty = 0.0
        penalty += traj['delta_violation'] * self.weights['delta']
        penalty += self._penalty_efl(efl)
        penalty += self._penalty_gaps(d1, d2, d3, CA1, CA2, CA3, CA4)
        penalty += self._penalty_petzval(f1_dyn, f2, f3, f4_dyn)
        penalty += self._penalty_monotonicity(z2, z3)
        penalty += self._penalty_smoothness(z3)
        penalty += self._penalty_ca1(CA1)
        penalty += self._penalty_root_center(m3)

        return penalty

    def optimize(self, callback=None, extra_seeds=None) -> bool:
        # 原始系数列表：[1.15, 1.25, 1.35, 1.45, 1.55, 1.65]
        # 为加速缩减为 3 个：1.2, 1.4, 1.6
        ttl_candidates = [self.config.ttl_target * r for r in [1.2, 1.4, 1.6]]

        if callback:
            callback(f"扫描 {len(ttl_candidates)} 个 TTL 候选值...")

        best_fun = float('inf')
        best_x = None
        best_ttl = self.config.ttl_target

        bounds = self._compute_bounds()  # 6维

        for ttl_val in ttl_candidates:
            t_G4 = self.system._t_G4
            self.system.z_G4_ref = ttl_val - self.config.bfd_target

            best_res_fun = float('inf')
            best_res_x = None
            seeds = [42, 123, 7]  # 默认三种子
            if extra_seeds:
                seeds.extend(extra_seeds)
            for s in seeds:
                res = differential_evolution(
                    self.objective_function, bounds,
                    strategy='best1bin', maxiter=120, popsize=20,
                    tol=0.005, seed=s, workers=1, disp=False
                )
                if res.fun < best_res_fun:
                    best_res_fun = res.fun
                    best_res_x = res.x.copy()

            ttl_dev = ((ttl_val - self.config.ttl_target) / self.config.ttl_target) ** 2
            total = best_res_fun + ttl_dev * 1.0e5

            if callback:
                callback(f"  TTL={ttl_val:.0f} -> {total:.2e}")

            if total < best_fun:
                best_fun = total
                best_x = best_res_x.copy()
                best_ttl = ttl_val

        # 精修
        self.system.z_G4_ref = best_ttl - self.config.bfd_target

        if callback:
            callback(f"最佳 TTL={best_ttl:.0f}，单纯形精修...")

        res_local = minimize(
            self.objective_function, best_x,
            method='Nelder-Mead',
            options=dict(xatol=1e-6, fatol=1e-6, maxiter=5000)
        )

        final_x = res_local.x if res_local.fun < best_fun else best_x
        self.best_params = final_x
        self.best_ttl = best_ttl

        f2, f3, m2_W, m2_T, f1_fac, f4_fac = final_x
        f1_dyn = self.config.f1 * f1_fac
        f4_dyn = self.config.f4 * f4_fac
        self.best_trajectory = self.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        return True

    def get_penalty_diagnostics(self, params: np.ndarray) -> dict:
        f2, f3, m2_W, m2_T, f1_fac, f4_fac = params

        if m2_W <= m2_T:
            return {"致命错误": "广角/长焦放大率倒置 (m2_W <= m2_T)"}

        f1_dyn = self.config.f1 * f1_fac
        f4_dyn = self.config.f4 * f4_fac

        traj = self.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

        z2, z3, efl, m3 = traj['z2'], traj['z3'], traj['efl'], traj['m3']
        d1, d2, d3 = traj['d1'], traj['d2'], traj['d3']
        CA1, CA2, CA3, CA4 = traj['CA1'], traj['CA2'], traj['CA3'], traj['CA4']

        p_delta = traj['delta_violation'] * self.weights['delta']
        p_efl = self._penalty_efl(efl)
        p_gaps = self._penalty_gaps(d1, d2, d3, CA1, CA2, CA3, CA4)
        p_petzval = self._penalty_petzval(f1_dyn, f2, f3, f4_dyn)
        p_mono = self._penalty_monotonicity(z2, z3)
        p_smooth = self._penalty_smoothness(z3)
        p_root = self._penalty_root_center(m3)
        p_ca1 = self._penalty_ca1(CA1)

        total = p_delta + p_efl + p_gaps + p_petzval + p_mono + p_smooth + p_root + p_ca1

        return {
            "总分 (Total)": total,
            "1. 数学越界 (Delta)": p_delta,
            "2. 焦距误差 (EFL)": p_efl,
            "3. 物理防撞 (Gaps)": p_gaps,
            "4. 场曲控制 (Petzval)": p_petzval,
            "5. G2单调性 (Mono)": p_mono,
            "6. 轨迹平滑 (Smooth)": p_smooth,
            "7. 换根失败 (Root)": p_root,
            "8. G1口径超标 (CA1)": p_ca1
        }

    # ── 辅助函数保持不变 ──
    def _penalty_efl(self, efl: np.ndarray) -> float:
        return (
            ((efl[0]  - self.config.f_wide) / self.config.f_wide) ** 2 +
            ((efl[-1] - self.config.f_tele) / self.config.f_tele) ** 2
        ) * self.weights['efl']

    def _penalty_gaps(self, d1, d2, d3, CA1, CA2, CA3, CA4) -> float:
        min_air_gap = 2.0
        penalty  = np.sum(np.maximum(0.0, min_air_gap - d1) ** 2)
        penalty += np.sum(np.maximum(0.0, min_air_gap - d2) ** 2)
        penalty += np.sum(np.maximum(0.0, min_air_gap - d3) ** 2)
        return penalty * self.weights['gaps']

    def _penalty_petzval(self, f1_dyn: float, f2: float, f3: float, f4_dyn: float) -> float:
        cfg = self.config
        P_sum = (
            1.0 / (f1_dyn * cfg.n_G1) +
            1.0 / (f2 * cfg.n_G2) +
            1.0 / (f3 * cfg.n_G3) +
            1.0 / (f4_dyn * cfg.n_G4)
        )
        return max(0.0, abs(P_sum) - 0.015) * self.weights['petzval']

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

    def _penalty_ca1(self, CA1: np.ndarray) -> float:
        violation = np.sum(np.maximum(0.0, CA1 - self.config.max_ca1))
        return violation * self.weights['ca1_limit']
