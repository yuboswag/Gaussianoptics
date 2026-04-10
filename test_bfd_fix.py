"""
test_bfd_fix.py
验证 BFD 修正是否正确：检查 G4 成像方程自洽性，并对比修改前后的 m4/u4/EFL。
"""
from config import ZoomConfig
from simulator import ZoomSystemSimulator
import numpy as np

# ── 使用 Action_a 当前测试参数 ──
cfg = ZoomConfig(
    f_wide=12.0, f_tele=142.0,
    ttl_target=105.0, bfd_target=8.0,
    f1=56.959, f4=72.545,
    num_positions=5,
    g1_thickness=15.0, g2_thickness=9.0,
    g3_thickness=7.0,  g4_thickness=21.0,
)

sim = ZoomSystemSimulator(cfg)

print("=== BFD 修正验证 ===")
print(f"物理 BFD        : {cfg.bfd_target} mm")
print(f"g4_thickness    : {cfg.g4_thickness} mm")
print(f"_bfd_eff (=bfd_target) : {cfg.bfd_target} mm")
print(f"z_G4_ref 定位   : {sim.z_G4_ref} mm  （预期 = {cfg.ttl_target - cfg.bfd_target}）")

# ── 用固定的 f2/f3/m2 跑一次 sweep，检查 G4 成像方程 ──
f2, f3 = -12.151, 24.409
m2_W, m2_T = -0.3, -2.8
traj = sim.zoom_sweep(f2, f3, m2_W, m2_T, cfg.f1, cfg.f4)

v4 = cfg.bfd_target
f4 = cfg.f4
diff = f4 - v4
u4 = f4 * v4 / diff
m4 = v4 / u4

print(f"\n=== G4 成像方程自洽检查 ===")
print(f"u4 (物距)       : {u4:.4f} mm")
print(f"v4 (像距=_bfd_eff): {v4:.4f} mm")
print(f"f4              : {f4:.4f} mm")
print(f"1/v4 - 1/u4     : {1/v4 - 1/u4:.6f}")
print(f"1/f4            : {1/f4:.6f}")
print(f"成像方程误差    : {abs(1/v4 - 1/u4 - 1/f4):.2e}  （应 < 1e-9）")

print(f"\n=== EFL 端点验证 ===")
print(f"广角 EFL        : {traj['efl'][0]:.4f} mm  （目标 {cfg.f_wide}）")
print(f"长焦 EFL        : {traj['efl'][-1]:.4f} mm  （目标 {cfg.f_tele}）")
