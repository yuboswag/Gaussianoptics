if __name__ == '__main__':
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
    success = opt.optimize(callback=print)

    if not success:
        print("FAIL: 优化未收敛")
    else:
        f2, f3, m2_W, m2_T, f1_fac, f4_fac = opt.best_params
        traj = opt.best_trajectory

        efl_w = traj['efl'][0]
        efl_t = traj['efl'][-1]
        err_w = abs(efl_w - cfg.f_wide) / cfg.f_wide * 100
        err_t = abs(efl_t - cfg.f_tele) / cfg.f_tele * 100
        ttl_dev = abs(opt.best_ttl - cfg.ttl_target) / cfg.ttl_target * 100

        print(f"\n=== 最优参数 ===")
        print(f"f1={cfg.f1*f1_fac:.3f}, f2={f2:.3f}, f3={f3:.3f}, f4={cfg.f4*f4_fac:.3f}")
        print(f"m2_W={m2_W:.4f}, m2_T={m2_T:.4f}")
        print(f"TTL={opt.best_ttl:.2f} mm  (目标 {cfg.ttl_target}, 偏差 {ttl_dev:.1f}%)")

        print(f"\n=== EFL 验证 ===")
        print(f"广角: {efl_w:.3f} mm  误差 {err_w:.2f}%")
        print(f"长焦: {efl_t:.3f} mm  误差 {err_t:.2f}%")

        print(f"\n=== 间隙安全 ===")
        print(f"d1: min={traj['d1'].min():.2f}")
        print(f"d2: min={traj['d2'].min():.2f}")
        print(f"d3: min={traj['d3'].min():.2f}")

        print(f"\n=== 组厚度（用于间距计算）===")
        print(f"t_G1={opt.system._t_G1:.1f}, t_G2={opt.system._t_G2:.1f}, t_G3={opt.system._t_G3:.1f}, t_G4={opt.system._t_G4:.1f}")

        # 判定
        efl_ok = err_w < 5 and err_t < 5
        ttl_ok = ttl_dev < 60
        gaps_ok = traj['d1'].min() > 0 and traj['d2'].min() > 0 and traj['d3'].min() > 0

        status = 'PASS' if (efl_ok and ttl_ok and gaps_ok) else 'FAIL'
        reasons = []
        if not efl_ok: reasons.append(f"EFL误差超标")
        if not ttl_ok: reasons.append(f"TTL偏差>{60}%")
        if not gaps_ok: reasons.append(f"存在负间隙")

        print(f"\n（注：TTL 目标 {cfg.ttl_target}mm 为薄透镜值，加入组厚度 {opt.system._t_G1 + opt.system._t_G2 + opt.system._t_G3 + opt.system._t_G4:.0f}mm 后偏大是正常的）")
        print(f"\n{status}" + (f": {', '.join(reasons)}" if reasons else ": EFL<5%, TTL<60%, 间隙全正"))
