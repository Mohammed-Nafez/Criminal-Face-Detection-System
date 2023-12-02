from cx_Freeze import setup, Executable
from GUI import Ui_Form1
setup(
    name="pythonProject2",
    version="1.0",
    description="My Python Application",
    executables=[Executable("loginGUI.py")]
)
