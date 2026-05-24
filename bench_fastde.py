import time
from optimizer import ZoomLensOptimizer
from simulator import ZoomConfig
from scipy.optimize import differential_evolution as _de_orig

# Monkey-patch DE: 强制精简参数
import optimizer as _opt_mod
import scipy.optimize as _sp_opt
def _de_fast(func, bounds, **kwargs):
    kwargs['maxiter'] = 60
    kwargs['popsize'] = 15
    kwargs['tol'] = 0.01
    return _de_orig(func, bounds, **kwargs)
_sp_opt.differential_evolution = _de_fast
_opt_mod.differential_evolution = _de_fast


def main():
    cfg = ZoomConfig(
        f_wide=11.9, f_tele=121.0, ttl_target=110.0, bfd_target=8.0,
        f1=55.0, f4=60.0, sensor_size=7.6,
        f_number=4.0, f_number_tele=5.6, max_ca1=40.0, num_positions=61,
        stop_shift=0.0, stop_group=2, vignetting=0.7, constant_f_number=False,
        g1_thickness=15.0, g2_thickness=9.0, g3_thickness=7.0, g4_thickness=21.0,
        n_eff_G1=1.65, n_eff_G2=1.70, n_eff_G3=1.65, n_eff_G4=1.65,
        v_eff_G1=50.0, v_eff_G2=30.0, v_eff_G3=50.0, v_eff_G4=50.0,
    )
    t0 = time.perf_counter()
    opt = ZoomLensOptimizer(cfg)
    opt.optimize(callback=None, extra_seeds=None)
    elapsed = time.perf_counter() - t0
    print(f"\n[FASTDE] 单轮耗时: {elapsed:.1f}s")
    print(f"[FASTDE] best_score: {opt.best_params is not None}")
    if opt.best_params is not None:
        bd = opt.get_penalty_diagnostics(opt.best_params)
        print(f"[FASTDE] 总分: {bd.get('总分 (Total)', 'N/A')}")
        for k, v in sorted(bd.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"[FASTDE]   {k}: {v:.2e}")


if __name__ == '__main__':
    main()