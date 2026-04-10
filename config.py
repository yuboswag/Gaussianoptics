"""
config.py
光学系统配置：字体设置与 ZoomConfig 数据类
"""

import warnings
import platform
import matplotlib
from dataclasses import dataclass
warnings.filterwarnings("ignore", module="matplotlib")

# ── 中文字体自动适配 ──────────────────────────────────────────
_os = platform.system()
if _os == 'Windows':
    matplotlib.rcParams.update({'font.sans-serif': ('Microsoft YaHei', 'SimHei', 'DejaVu Sans')})
elif _os == 'Darwin':
    matplotlib.rcParams.update({'font.sans-serif': ('PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans')})
else:
    matplotlib.rcParams.update({'font.sans-serif': ('WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans')})
matplotlib.rcParams.update({'axes.unicode_minus': False})


@dataclass
class ZoomConfig:
    f_wide: float = 10.0
    f_tele: float = 100.0
    bfd_target: float = 15.0
    ttl_target: float = 180.0
    f1: float = 100.0
    f4: float = 45.0
    num_positions: int = 61
    # 工程级物理参数
    sensor_size: float = 11.0   # 像面直径(mm)，约 2/3英寸~1英寸传感器对角线
    f_number: float = 4.0       # 广角端 F 数
    f_number_tele: float = 5.6  # 长焦端 F 数
    stop_shift: float = 0.0     # 光阑相对于所在组元的偏移量（正值=向像方）
    stop_group: int = 3         # 光阑所在组元编号 (1=G1, 2=G2, 3=G3, 4=G4)
    constant_f_number: bool = True
    max_ca1: float = 50.0
    # 渐晕控制
    vignetting: float = 0.5
    vignetting_tele: float = 0.8  # 长焦端渐晕系数
    # 各组元轴向玻璃厚度估算（用于 BFD/TTL 修正，可在 GUI 中调整）
    g1_thickness: float = 15.0   # G1 前固定组
    g2_thickness: float = 9.0    # G2 变倍组
    g3_thickness: float = 7.0    # G3 补偿组
    g4_thickness: float = 21.0   # G4 后固定组
    # 各组总厚度（玻璃+组内间隔之和，mm），用于主平面→实际空气间隔转换
    t_G1: float = 0.0
    t_G2: float = 0.0
    t_G3: float = 0.0
    t_G4: float = 0.0