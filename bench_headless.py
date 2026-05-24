"""Headless benchmark: 直接调 ZoomLensOptimizer，无 Gradio、无 threading、无 queue."""
import time
from optimizer import ZoomLensOptimizer
from simulator import ZoomConfig


def main():
    cfg = ZoomConfig(
        f_wide=11.9, f_tele=121.0,
        ttl_target=110.0, bfd_target=8.0,
        f1=55.0, f4=60.0,
        sensor_size=7.6,
        f_number=4.0, f_number_tele=5.6,
        max_ca1=40.0, num_positions=61,
        stop_shift=0.0, stop_group=2,
        vignetting=0.7, constant_f_number=False,
        g1_thickness=15.0, g2_thickness=9.0,
        g3_thickness=7.0, g4_thickness=21.0,
        n_eff_G1=1.65, n_eff_G2=1.70,
        n_eff_G3=1.65, n_eff_G4=1.65,
        v_eff_G1=50.0, v_eff_G2=30.0,
        v_eff_G3=50.0, v_eff_G4=50.0,
    )

    print(">>> Headless benchmark 开始 (1 轮，callback=None)")
    t0 = time.perf_counter()
    opt = ZoomLensOptimizer(cfg)
    opt.optimize(callback=None, extra_seeds=None)
    elapsed = time.perf_counter() - t0
    print(f"\n[BENCH] headless 单轮耗时: {elapsed:.1f}s")
    print(f"[BENCH] best_score found: {opt.best_params is not None}")


if __name__ == '__main__':
    main()