import os
import shutil
import tkinter as tk
from tkinter import filedialog

# Set the path to the directory where you want to move the image
destination_dir = r"C:\Users\Mohammd Nafez Aloul\PycharmProjects\pythonProject2\images"

# Create a GUI window to select the image file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Check if a file was selected
if file_path:
    # Get the filename from the path
    filename = os.path.basename(file_path)

    # Move the file to the destination directory
    shutil.move(file_path, os.path.join(destination_dir, filename))
    print("Image moved successfully!")
else:
    print("No file selected!")
