"""
main.py
=======
Entry point for the CPT clock performance model.

Usage:
    python main.py

Requirements: numpy, scipy, matplotlib  (see requirements.txt)
"""

import tkinter as tk
from gui import CPTClockApp


def main():
    root = tk.Tk()
    app  = CPTClockApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
