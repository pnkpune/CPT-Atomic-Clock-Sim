#include <windows.h>
#include <stdio.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    
    ZeroMemory(&pi, sizeof(pi));

    // Point to MSYS2 UCRT64 Python environment without a console
    char cmd[] = "C:\\msys64\\ucrt64\\bin\\pythonw.exe main.py";

    // Attempt to start the process using MSYS2's Python first
    if (!CreateProcess(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        
        // Fallback: If not found, try the generic Windows PATH
        char cmd2[] = "pythonw.exe main.py";
        if (!CreateProcess(NULL, cmd2, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
            
            // Neither could launch, most likely missing python payload or main script
            MessageBox(NULL, 
                       "Failed to launch 'main.py'.\n"
                       "Please ensure MSYS2/Python is installed and 'main.py' is in the same directory.", 
                       "CPT Clock Launcher Error", 
                       MB_OK | MB_ICONERROR);
            return 1;
        }
    }

    // Successfully launched Python in background, don't wait for completion and exit
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    return 0;
}
