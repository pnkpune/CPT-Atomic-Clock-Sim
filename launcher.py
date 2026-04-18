import subprocess
import sys
import os
import tkinter as tk
from tkinter import messagebox

if __name__ == '__main__':
    # Location of the actual physics script
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(__file__)
        
    target_script = os.path.join(exe_dir, "main.py")
    
    # Try using MSYS2 ucrt64 pythonw if it exists to ensure standard environment matches
    msys_python = r"C:\msys64\ucrt64\bin\pythonw.exe"
    
    # If not found next to the executable (e.g., inside dist/), check parent folder
    if not os.path.exists(target_script):
        parent_dir = os.path.dirname(exe_dir)
        target_script = os.path.join(parent_dir, "main.py")
        
    if not os.path.exists(target_script):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to locate main.py.\nPlease place this executable next to main.py.")
        sys.exit(1)

    try:
        # Popen without waiting, so the launcher closes immediately
        # 0x08000000 is windows CREATE_NO_WINDOW
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        
        if os.path.exists(msys_python):
            subprocess.Popen([msys_python, target_script], creationflags=creation_flags)
        else:
            subprocess.Popen(["pythonw", target_script], creationflags=creation_flags)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Execution Error", f"Failed to launch Python:\n{str(e)}")
        sys.exit(1)
