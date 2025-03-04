import tkinter as tk
import subprocess
from tkinter import PhotoImage
from PIL import Image, ImageTk

def run_model(script_name):
    root.withdraw()  # Hide the main window while the model runs
    try:
        subprocess.run(['python', script_name])  # Run the selected script
    except Exception as e:
        print(f"Error running {script_name}: {e}")
    root.deiconify()  # Show the main window again when the script finishes

root = tk.Tk()
root.title("Exercise Feedback System")
root.geometry("500x550")

# Create a smooth vertical gradient background
gradient_canvas = tk.Canvas(root, width=500, height=550, highlightthickness=0)
gradient_canvas.pack(fill="both", expand=True)
for i in range(550):
    red = max(0, min(255, 255 - (i // 3)))
    green = max(0, min(255, 100 + (i // 6)))
    blue = max(0, min(255, 50 + (i // 2)))
    color = f"#{red:02x}{green:02x}{blue:02x}"
    gradient_canvas.create_line(0, i, 500, i, fill=color)

# Add title label with transparent background
gradient_canvas.create_text(250, 40, text="Exercise Feedback", font=("Brush Script MT", 28, "bold"), fill="black")

# Load and resize images
image_files = {
    "Bicep Curls": "images/bicep_curls.png",
    "Posture": "images/posture.png",
    "Squats": "images/squats.png",
    "Wrist Rotation": "images/wrist_rotation.png",
    "Lunges": "images/lunges.png",
    "Jumping Jacks": "images/jumping_jacks.png",
    "Exit": "images/exit.png"
}

icons = {}
for exercise, file in image_files.items():
    img = Image.open(file)
    img = img.resize((30, 30), Image.Resampling.LANCZOS)
    icons[exercise] = ImageTk.PhotoImage(img)

# Buttons for different exercises with images
exercises = {
    "Bicep Curls": "bicep_curls.py",
    "Posture": "posture.py",
    "Squats": "squats.py",
    "Wrist Rotation": "wrist_rotation.py",
    "Lunges": "lunges.py",
    "Jumping Jacks": "jumping_jacks.py"
}

button_colors = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#66ccff", "#ff66b2"]

# Adjusted positions with spacing
positions = [(130, 150), (130, 250), (130, 350), (370, 150), (370, 250), (370, 350)]

for index, ((exercise, script), (x, y)) in enumerate(zip(exercises.items(), positions)):
    button = tk.Button(
        root, text=exercise, command=lambda s=script: run_model(s), 
        image=icons[exercise], compound="left", padx=10,
        bg=button_colors[index], font=("Arial", 12, "bold"), width=200
    )
    gradient_canvas.create_window(x, y, window=button)

# Exit button (centered at the bottom)
exit_button = tk.Button(root, text="Exit", command=root.quit, image=icons["Exit"], compound="left", 
                        padx=10, bg="red", fg="white", font=("Arial", 12, "bold"))
gradient_canvas.create_window(250, 500, window=exit_button)

root.mainloop()
