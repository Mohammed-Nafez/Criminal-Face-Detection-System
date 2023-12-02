import subprocess
import sys
import os

# Specify the path to the directory you want to open
directory_path = r'C:\Users\Mohammd Nafez Aloul\PycharmProjects\pythonProject2\Matches'

# Get the current operating system
operating_system = sys.platform

try:
    if operating_system.startswith('win'):  # Windows
        subprocess.Popen(['explorer', directory_path])
    elif operating_system.startswith('darwin'):  # macOS
        subprocess.Popen(['open', directory_path])
    elif operating_system.startswith('linux'):  # Linux
        subprocess.Popen(['xdg-open', directory_path])
    else:
        print("Unsupported operating system.")
    
except FileNotFoundError:
    print("Directory not found.")
    
except NotADirectoryError:
    print("Path specified is not a directory.")
    
except PermissionError:
    print("Permission denied to open the directory.")
