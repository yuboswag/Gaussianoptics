from config import ZoomConfig
from optimizer import ZoomLensOptimizer
import numpy as np

cfg = ZoomConfig(
    f_wide=12.0, f_tele=142.0,
    ttl_target=105.0, bfd_target=8.0,
    f1=56.959, f4=72.545,
    num_positions=21,
    g1_thickness=15.0, g2_thickness=9.0,
    g3_thickness=7.0, g4_thickness=21.0,
)

opt = ZoomLensOptimizer(cfg)

# 诊断当前收敛到的"坏解"
bad_params = np.array([-19.700, 35.074, -0.7812, -10.7594,
                        45.932/cfg.f1, 60.373/cfg.f4])
print("=== Bad Solution Penalty ===")
diag = opt.get_penalty_diagnostics(bad_params)
for k, v in diag.items():
    print(f"  {k}: {v:.2e}")

# 对比：用接近目标的"好参数"
good_params = np.array([-12.151, 24.409, -0.30, -2.80, 1.0, 1.0])
print("\n=== Reference Solution Penalty ===")
diag2 = opt.get_penalty_diagnostics(good_params)
for k, v in diag2.items():
    print(f"  {k}: {v:.2e}")

# 诊断参考解的间隙
good_params = np.array([-12.151, 24.409, -0.30, -2.80, 1.0, 1.0])
f2, f3, m2_W, m2_T, f1_fac, f4_fac = good_params
f1_dyn = cfg.f1 * f1_fac
f4_dyn = cfg.f4 * f4_fac
traj = opt.system.zoom_sweep(f2, f3, m2_W, m2_T, f1_dyn, f4_dyn)

print("\n=== 参考解间隙诊断 ===")
print(f"d1: min={traj['d1'].min():.2f}  max={traj['d1'].max():.2f}")
print(f"d2: min={traj['d2'].min():.2f}  max={traj['d2'].max():.2f}")
print(f"d3: min={traj['d3'].min():.2f}  max={traj['d3'].max():.2f}")
print(f"EFL wide={traj['efl'][0]:.3f}  tele={traj['efl'][-1]:.3f}")
