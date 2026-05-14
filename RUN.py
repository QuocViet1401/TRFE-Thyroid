
import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    clear_screen()
    print("       TRFEPLUS Thyroid Ultrasound Analysis System")
    print("\n   [1] Complete Setup (Install + Train + Run App)")
    print("   [2] Train Model Only")
    print("   [3] Run Web App Only")
    print("   [4] Exit")
    return input("Enter your choice [1-4]: ").strip()

def run_command(cmd, description="", cwd=None):
    if description:
        print(f"\n{description}...")
    try:
        if cwd is None:
            cwd = os.getcwd()
        result = subprocess.run(cmd, shell=True, cwd=cwd)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def setup_venv():
    print("\nChecking Python...")
    if not run_command("python --version"):
        print("ERROR: Python not installed!")
        return False
    
    if not Path(".venv").exists():
        print("Creating virtual environment...")
        if not run_command("python -m venv .venv"):
            print("ERROR: Failed to create venv!")
            return False
    
    print("Installing dependencies...")
    if os.name == 'nt':
        python_exe = r".venv\Scripts\python.exe"
        cmd = f'{python_exe} -m pip install -q -r requirements.txt'
    else:
        python_exe = ".venv/bin/python"
        cmd = f'{python_exe} -m pip install -q -r requirements.txt'
    
    return run_command(cmd)

def option1():
    """Complete Setup"""
    clear_screen()
    print("                COMPLETE SETUP")
    
    if not setup_venv():
        print("Setup failed!")
        input("Press Enter to continue...")
        return
    
    if os.name == 'nt':
        python_exe = r".venv\Scripts\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    print("\nTraining model...")
    if not run_command(f'{python_exe} pretrain_model.py'):
        print("Training failed!")
        input("Press Enter to continue...")
        return
    
    model_path = Path("run/trfeplus/fold0/trfeplus_best.pth")
    if not model_path.exists():
        print("Model file not found!")
        input("Press Enter to continue...")
        return
    
    print("\nStarting web app...")
    print("Opening http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    run_command(f'{python_exe} -m streamlit run app.py')

def option2():
    """Train Only"""
    clear_screen()
    print("                TRAINING MODEL")
    
    if not Path(".venv").exists():
        print("Virtual environment not found! Run option [1] first.")
        input("Press Enter to continue...")
        return
    
    if os.name == 'nt':
        python_exe = r".venv\Scripts\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    run_command(f'{python_exe} pretrain_model.py')
    input("Press Enter to continue...")

def option3():
    """Run App Only"""
    clear_screen()
    print("                STARTING WEB APP")
    
    if not Path(".venv").exists():
        print("Virtual environment not found! Run option [1] first.")
        input("Press Enter to continue...")
        return
    
    model_path = Path("run/trfeplus/fold0/trfeplus_best.pth")
    if not model_path.exists():
        print("ERROR: Model not found!")
        print("Please run option [2] to train first.")
        input("Press Enter to continue...")
        return
    
    if os.name == 'nt':
        python_exe = r".venv\Scripts\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    print("Opening http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    run_command(f'{python_exe} -m streamlit run app.py')

def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            option1()
        elif choice == "2":
            option2()
        elif choice == "3":
            option3()
        elif choice == "4":
            print("\nExiting...")
            sys.exit(0)
        else:
            print("Invalid choice!")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        sys.exit(0)
