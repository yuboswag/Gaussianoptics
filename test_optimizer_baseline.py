"""
test_optimizer_baseline.py
还原 BFD 修改后的基线验证，确认优化器正常收敛。
"""
from config import ZoomConfig
from optimizer import ZoomLensOptimizer
import numpy as np

cfg = ZoomConfig(
    f_wide=12.0, f_tele=142.0,
    ttl_target=105.0, bfd_target=8.0,
    f1=56.959, f4=72.545,
    num_positions=21,
    g1_thickness=15.0, g2_thickness=9.0,
    g3_thickness=7.0,  g4_thickness=21.0,
)

if __name__ == '__main__':
    print("开始基线优化...")
    opt = ZoomLensOptimizer(cfg)
    success = opt.optimize(callback=print)

    if not success:
        print("FAIL: 优化未收敛")
    else:
        traj = opt.best_trajectory
        efl_w = traj['efl'][0]
        efl_t = traj['efl'][-1]
        err_w = abs(efl_w - cfg.f_wide)  / cfg.f_wide  * 100
        err_t = abs(efl_t - cfg.f_tele) / cfg.f_tele * 100

        print(f"\n=== 优化结果 ===")
        print(f"广角 EFL : {efl_w:.3f} mm  目标 {cfg.f_wide}  误差 {err_w:.1f}%")
        print(f"长焦 EFL : {efl_t:.3f} mm  目标 {cfg.f_tele}  误差 {err_t:.2f}%")
        print(f"TTL : {opt.best_ttl:.2f} mm  目标 {cfg.ttl_target}")

        f2, f3, m2_W, m2_T, f1_fac, f4_fac = opt.best_params
        print(f"\n=== 最优参数 ===")
        print(f"f2={f2:.3f}, f3={f3:.3f}")
        print(f"m2_W={m2_W:.4f}, m2_T={m2_T:.4f}")
        print(f"f1={cfg.f1*f1_fac:.3f}, f4={cfg.f4*f4_fac:.3f}")

        print(f"\n=== 间隙诊断 ===")
        print(f"d1: min={traj['d1'].min():.2f}  max={traj['d1'].max():.2f}")
        print(f"d2: min={traj['d2'].min():.2f}  max={traj['d2'].max():.2f}")
        print(f"d3: min={traj['d3'].min():.2f}  max={traj['d3'].max():.2f}")

        ok_w = err_w < 5.0
        ok_t = err_t < 5.0
        status = 'PASS' if ok_w and ok_t else 'FAIL'
        print(f"\n{status}: EFL 端点误差 广角{err_w:.1f}% / 长焦{err_t:.2f}%")
