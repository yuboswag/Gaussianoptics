"""
main.py
程序入口：启动变焦镜头设计工具
"""

import tkinter as tk
from gui import ZoomLensDesignerGUI


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    ZoomLensDesignerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
