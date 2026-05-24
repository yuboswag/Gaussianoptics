"""
simulator.py
光学算法引擎：四组元变焦系统近轴追迹与运动学仿真 (支持 f1/f4 动态浮动)
"""

import numpy as np
from config import ZoomConfig


class ZoomSystemSimulator:
    def __init__(self, config: ZoomConfig):
        self.config = config
        self._t_G1 = config.t_G1 if config.t_G1 > 0 else config.g1_thickness
        self._t_G2 = config.t_G2 if config.t_G2 > 0 else config.g2_thickness
        self._t_G3 = config.t_G3 if config.t_G3 > 0 else config.g3_thickness
        self._t_G4 = config.t_G4 if config.t_G4 > 0 else config.g4_thickness
        self.z_G1 = self._t_G1
        self.z_G4_ref = config.ttl_target - config.bfd_target
        self.bfd_override = None  # None = use config.bfd_target

    def zoom_sweep(self, f2: float, f3: float, m2_W: float, m2_T: float, f1_dyn: float, f4_dyn: float) -> dict | None:
        N = self.config.num_positions
        m2_arr = np.linspace(m2_W, m2_T, N)

        f1 = f1_dyn
        f4 = f4_dyn

        # G4 位置：由外层 z_G4_ref 决定（TTL 扫描更新）
        z_G4 = self.z_G4_ref

        # G4 物像关系：固定 BFD
        bfd = self.bfd_override if self.bfd_override is not None else self.config.bfd_target
        v4 = bfd
        diff = f4 - v4
        if abs(diff) < 1e-5:
            f4 += 2e-5 if diff >= 0 else -2e-5
            diff = f4 - v4
        u4 = f4 * v4 / diff
        m4 = v4 / u4

        z_G4_obj = z_G4 + u4

        # --- 运动学求解 ---
        z2, z2_img = self._g2_kinematics(f1, f2, m2_arr)
        z3, m3_arr, root1, root2, delta_violation = self._g3_kinematics(f3, z2_img, z_G4_obj, m2_arr, N)

        efl_arr = f1 * np.abs(m2_arr * m3_arr * m4)
        z3_root1 = z2_img - f3 * (1.0 / root1 - 1.0)
        z3_root2 = z2_img - f3 * (1.0 / root2 - 1.0)

        # 薄透镜主平面间距
        d1_thin = z2 - self.z_G1
        d2_thin = z3 - z2
        d3_thin = z_G4 - z3

        # 转换为实际空气间隔（半厚度近似：各组主平面在组中心）
        d1 = d1_thin - (self._t_G1 + self._t_G2) / 2  # G1-G2 间隔：扣除 G1、G2 各半厚度
        d2 = d2_thin - (self._t_G2 + self._t_G3) / 2  # G2-G3 间隔：扣除 G2、G3 各半厚度
        # G3-G4 间距：扣除 G3 半厚度 + G4 半厚度 + G4 后主面偏移修正
        # delta_Hp_G4 > 0 表示 G4 主平面偏向像方，意味着与 G3 之间的实际气隙更小
        d3 = d3_thin - (self._t_G3 + self._t_G4) / 2 - self.config.delta_Hp_G4

        # _paraxial_ca 现在额外返回三组边缘光线追迹数组：
        #   h_m     — 各组元入射处边缘光线高度，shape (N, 4)
        #   u_in_m  — 各组元折射前边缘光线角度，shape (N, 4)
        #   u_out_m — 各组元折射后边缘光线角度，shape (N, 4)
        # 近轴光线追迹用主面间距 d_thin（薄透镜模型的标准做法）
        # d1/d2/d3 是实际空气间隔，仅用于物理防撞惩罚和导出，不参与追迹
        CA1, CA2, CA3, CA4, h_m, u_in_m, u_out_m = self._paraxial_ca(
            f1, f2, f3, f4, d1_thin, d2_thin, d3_thin, efl_arr, N
        )

        return dict(
            f1=f1, f2=f2, f3=f3, f4=f4,  # 记录当前使用的焦距
            z2=z2, z3=z3, m3=m3_arr, m2=m2_arr, efl=efl_arr,
            root1_z3=z3_root1, root2_z3=z3_root2,
            ttl=np.full(N, self.config.ttl_target),
            # 薄透镜 TTL 与物理 TTL（新增诊断，不改优化逻辑）
            thin_ttl=np.full(N, self.z_G4_ref + bfd),
            phys_ttl=np.full(N, self.z_G4_ref + bfd + (self._t_G1 + self._t_G4) / 2.0),
            CA1=CA1, CA2=CA2, CA3=CA3, CA4=CA4,
            d1=d1, d2=d2, d3=d3,  # 实际空气间隔（已扣除组厚度）
            d1_thin=d1_thin, d2_thin=d2_thin, d3_thin=d3_thin,  # 薄透镜主平面间距（供诊断）
            bfd=bfd,
            delta_violation=delta_violation,
            # ── 新增：边缘光线追迹数据（列顺序：G1, G2, G3, G4）──────
            h_m=h_m,       # 各组元的边缘光线高度 (mm)
            u_in_m=u_in_m,  # 各组元折射前的边缘光线角度 (rad)
            u_out_m=u_out_m, # 各组元折射后的边缘光线角度 (rad)
        )

    def _g2_kinematics(self, f1: float, f2: float, m2_arr: np.ndarray):
        u2 = f2 * (1.0 / m2_arr - 1.0)
        v2 = f2 * (1.0 - m2_arr)
        z_G1_img = self.z_G1 + f1
        z2 = z_G1_img - u2
        z2_img = z2 + v2
        return z2, z2_img

    def _g3_kinematics(self, f3: float, z2_img: np.ndarray, z_G4_obj: float, m2_arr: np.ndarray, N: int):
        L_avail = z_G4_obj - z2_img
        
        delta_orig = L_avail * (L_avail - 4.0 * f3)
        delta_violation = np.sum(np.maximum(0.0, -delta_orig))
        delta = np.maximum(0.0, delta_orig)
        sqrt_delta = np.sqrt(delta)

        root1 = (2.0 * f3 - L_avail - sqrt_delta) / (2.0 * f3)
        root2 = (2.0 * f3 - L_avail + sqrt_delta) / (2.0 * f3)

        cross_idx = np.argmin(delta)
        opts = [
            root1,
            root2,
            np.concatenate((root1[:cross_idx], root2[cross_idx:])),
            np.concatenate((root2[:cross_idx], root1[cross_idx:]))
        ]
        
        best_m3 = opts[0]
        best_z3 = z2_img - f3 * (1.0 / best_m3 - 1.0)
        best_smooth = float('inf')
        
        for m3_cand in opts:
            z3_cand = z2_img - f3 * (1.0 / m3_cand - 1.0)
            smooth = np.sum(np.diff(z3_cand, n=2) ** 2)
            if smooth < best_smooth:
                best_smooth = smooth
                best_m3 = m3_cand
                best_z3 = z3_cand

        return best_z3, best_m3, root1, root2, delta_violation

    def _trace_chief_ray(self, f1, f2, f3, f4, d1, d2, d3, N):
        """
        根据 config.stop_group 和 config.stop_shift 正向求解主光线。

        光阑位于 G_{sg} 主面向像方偏移 ss 处（y_stop = 0）。
        对每个光阑位置推导解析公式：从光阑出发正向追迹到像面，
        令 y_image = H_img 求出 u0，再反向追迹各组元高度。

        所有计算均向量化，d1/d2/d3 为 (N,) 数组。
        返回: y1_c, y2_c, y3_c, y4_c  各形状 (N,)
        """
        H_img = self.config.sensor_size / 2.0
        bfd   = self.bfd_override if self.bfd_override is not None else self.config.bfd_target
        sg    = getattr(self.config, 'stop_group', 3)
        ss    = self.config.stop_shift  # 正值 = 向像方偏移

        if sg == 1:
            # ── 光阑在 G1 主面向像方 ss 处（G1-G2 空间内）────────────
            # 从光阑出发经 G2→G3→G4 到像面，推导 u0
            L1 = d1 - ss               # 光阑到 G2 的距离
            A  = 1.0 - L1 / f2        # G2 折射后: u2' = A·u0
            C  = L1 + d2 * A          # G3 处高度: y3 = C·u0
            B  = A  - C  / f3         # G3 折射后: u3' = B·u0
            Q  = C  + d3 * B          # G4 处高度: y4 = Q·u0
            D  = B  - Q  / f4         # G4 折射后: u4' = D·u0
            denom = Q + bfd * D
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            u0   = H_img / denom
            # 各组元主光线高度
            y1_c = -ss * u0            # y_G1 = y_stop - ss·u0 = -ss·u0
            y2_c = L1 * u0
            y3_c = C  * u0
            y4_c = Q  * u0

        elif sg == 2:
            # ── 光阑在 G2 主面向像方 ss 处（G2-G3 空间内）────────────
            # 从光阑出发经 G3→G4 到像面，推导 u0
            L2 = d2 - ss               # 光阑到 G3 的距离
            B  = 1.0 - L2 / f3        # G3 折射后: u3' = B·u0
            C  = L2 + d3 * B          # G4 处高度: y4 = C·u0
            D  = B  - C  / f4         # G4 折射后: u4' = D·u0
            denom = C + bfd * D
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            u0   = H_img / denom
            # 各组元主光线高度
            y2_c = -ss * u0            # y_G2 = -ss·u0
            y3_c = L2 * u0
            y4_c = C  * u0
            # 反向追迹到 G1
            u2_in = u0 * (1.0 - ss / f2)   # G2 物侧角
            y1_c  = y2_c - d1 * u2_in

        elif sg == 3:
            # ── 光阑在 G3 主面向像方 ss 处（G3-G4 空间内，原始逻辑）──
            # 从光阑出发经 G4 到像面，推导 u0（数学上与原代码完全等价）
            L3    = d3 - ss            # 光阑到 G4 的距离
            denom = L3 + bfd * (1.0 - L3 / f4)
            denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
            u0   = H_img / denom
            # 各组元主光线高度
            y3_c  = -ss * u0           # y_G3 = -ss·u0
            y4_c  = L3 * u0
            # 反向追迹到 G2, G1
            u3_in = u0 * (1.0 - ss / f3)
            y2_c  = y3_c - d2 * u3_in
            u2_in = u3_in + y2_c / f2
            y1_c  = y2_c - d1 * u2_in

        else:
            # ── 光阑在 G4 主面向像方 ss 处（G4-像面空间内）────────────
            # 光阑到像面有效距离 bfd - ss
            eff_bfd = bfd - ss
            if abs(eff_bfd) < 1e-9:
                eff_bfd = 1e-9
            u4_out = H_img / eff_bfd   # G4 折射后主光线角
            y4_c   = -ss * u4_out      # G4 处主光线高度
            # 反向追迹到 G3, G2, G1
            u4_in  = u4_out + y4_c / f4
            u3_out = u4_in             # G3-G4 空间中的主光线角
            y3_c   = y4_c - d3 * u3_out
            u3_in  = u3_out + y3_c / f3
            y2_c   = y3_c - d2 * u3_in
            u2_in  = u3_in  + y2_c / f2
            y1_c   = y2_c - d1 * u2_in

        return y1_c, y2_c, y3_c, y4_c

    def _paraxial_ca(self, f1, f2, f3, f4, d1, d2, d3, efl_arr, N):
        if self.config.constant_f_number:
            current_f_number = np.full(N, self.config.f_number)
        else:
            F_w = self.config.f_number
            F_t = self.config.f_number_tele
            ratio_arr = np.linspace(0.0, 1.0, N)
            current_f_number = F_w + (F_t - F_w) * ratio_arr

        # ── 边缘光线高度与折射角追迹 ──────────────────────────────
        # 约定：y_m 为各组元主面处的边缘光线高度；
        #        u_m_prime 为经该组元折射后（出射）的角度，
        #        因此 u_in[i] = u_m_prime[i-1]（前组出射 = 本组入射）。
        #
        # G1 入射为平行光：u_in_G1 = 0，出射角 u1_m' = -h/f1
        y1_m = efl_arr / (2.0 * current_f_number)  # G1 处边缘光高

        u1_m_prime = -y1_m / f1          # G1 出射角（即 G2 入射角）
        y2_m = y1_m + d1 * u1_m_prime    # G2 处边缘光高
        u2_m_prime = u1_m_prime - y2_m / f2  # G2 出射角
        y3_m = y2_m + d2 * u2_m_prime    # G3 处边缘光高
        u3_m_prime = u2_m_prime - y3_m / f3  # G3 出射角
        y4_m = y3_m + d3 * u3_m_prime    # G4 处边缘光高
        u4_m_prime = u3_m_prime - y4_m / f4  # G4 出射角

        # ── 打包为 (N, 4) 数组，列顺序：G1, G2, G3, G4 ───────────
        h_m     = np.column_stack([y1_m, y2_m, y3_m, y4_m])
        u_in_m  = np.column_stack([np.zeros(N), u1_m_prime, u2_m_prime, u3_m_prime])
        u_out_m = np.column_stack([u1_m_prime,  u2_m_prime, u3_m_prime, u4_m_prime])

        # ── 主光线追迹（支持任意光阑组元）────────────────────────
        # 根据 config.stop_group 和 config.stop_shift 计算各组元主光线高度
        y1_c, y2_c, y3_c, y4_c = self._trace_chief_ray(
            f1, f2, f3, f4, d1, d2, d3, N
        )

        # ── 渐晕系数线性插值 ──────────────────────────────────────
        vig_wide_val = getattr(self.config, 'vignetting',      0.5)
        vig_tele_val = getattr(self.config, 'vignetting_tele', 0.8)
        ratio = np.linspace(0.0, 1.0, N)
        V_arr = vig_wide_val + (vig_tele_val - vig_wide_val) * ratio

        # ── 各组元通光口径 ────────────────────────────────────────
        CA1_on_axis  = 2.0 * np.abs(y1_m)
        CA1_off_axis = 2.0 * (np.abs(y1_c) + V_arr * np.abs(y1_m))
        CA1 = np.maximum(CA1_on_axis, CA1_off_axis)

        CA2 = np.maximum(2.0 * np.abs(y2_m), 2.0 * (np.abs(y2_c) + V_arr * np.abs(y2_m)))
        CA3 = np.maximum(2.0 * np.abs(y3_m), 2.0 * (np.abs(y3_c) + V_arr * np.abs(y3_m)))
        CA4 = np.maximum(2.0 * np.abs(y4_m), 2.0 * (np.abs(y4_c) + V_arr * np.abs(y4_m)))

        return CA1, CA2, CA3, CA4, h_m, u_in_m, u_out_m