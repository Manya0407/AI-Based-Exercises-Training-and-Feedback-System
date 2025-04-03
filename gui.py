import tkinter as tk
import subprocess
from PIL import Image, ImageTk
import importlib.util
import sys
from model_handler import load_model
import traceback

def run_model(script_name):
    root.withdraw()
    try:
        # Ensure model is loaded before running any exercise
        if not load_model():
            print("Failed to load model. Please check your internet connection and try again.")
            root.deiconify()
            return
            
        # Import the exercise module dynamically
        spec = importlib.util.spec_from_file_location(script_name[:-3], script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function name by removing .py and converting to lowercase
        base_name = script_name[:-3].lower()
        
        # Define possible function names based on the script
        if script_name == "Lunges.py":
            possible_functions = ["run_lunges"]
        elif script_name == "WristRotation.py":
            possible_functions = ["run_wrist_rotation"]
        elif script_name == "BicepCurls.py":
            possible_functions = ["run_bicep_curls"]
        else:
            possible_functions = [
                f"run_{base_name}",
                f"run_{base_name.replace('_', '')}",
                base_name
            ]
        
        # Try each possible function name
        for function_name in possible_functions:
            if hasattr(module, function_name):
                result = getattr(module, function_name)()
                if result is False:  # If the function explicitly returns False
                    root.quit()  # Quit the entire application
                return
        
        print(f"Error: Could not find any of the following functions in {script_name}:")
        for function_name in possible_functions:
            print(f"  - {function_name}")
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        print(traceback.format_exc())  # Add this line to see the full error trace
    finally:
        root.deiconify()  # Always show the GUI again

def exit_fullscreen(event=None):
    root.attributes('-fullscreen', False)

def resize(event=None):
    """ Redraws everything smoothly when the window resizes """
    global screen_width, screen_height, cached_icon_size, cached_font_size

    screen_width, screen_height = root.winfo_width(), root.winfo_height()

    # Clear old gradient and redraw
    gradient_canvas.delete("all")
    for i in range(screen_height):
        color = f"#{max(0, min(255, 255 - (i // 3))):02x}{max(0, min(255, 100 + (i // 6))):02x}{max(0, min(255, 50 + (i // 2))):02x}"
        gradient_canvas.create_line(0, i, screen_width, i, fill=color)

    # Reposition title text
    title_font_size = max(20, int(screen_width * 0.04))
    gradient_canvas.create_text(screen_width // 2, int(screen_height * 0.1),
                                text="Exercise Feedback", font=("Brush Script MT", title_font_size, "bold"),
                                fill="black", tags="title")

    # Resize images dynamically
    new_icon_size = max(30, int(screen_width * 0.05))
    new_font_size = max(12, int(screen_width * 0.02))
    
    if new_icon_size != cached_icon_size or new_font_size != cached_font_size:
        cached_icon_size = new_icon_size
        cached_font_size = new_font_size
        
        for exercise, file in image_files.items():
            img = Image.open(file)
            img = img.resize((cached_icon_size, cached_icon_size), Image.Resampling.LANCZOS)
            icons[exercise] = ImageTk.PhotoImage(img)

    # Adjust button sizes and positions
    button_width = int(screen_width * 0.25)
    button_height = int(screen_height * 0.08)
    
    positions = [
        (screen_width * 0.3, screen_height * 0.3),
        (screen_width * 0.3, screen_height * 0.5),
        (screen_width * 0.3, screen_height * 0.7),
        (screen_width * 0.7, screen_height * 0.3),
        (screen_width * 0.7, screen_height * 0.5),
        (screen_width * 0.7, screen_height * 0.7),
    ]

    for i, (exercise, button) in enumerate(buttons.items()):
        x, y = positions[i]
        button.config(width=button_width, height=button_height, font=("Arial", cached_font_size, "bold"),
                      padx=10, pady=5, image=icons[exercise], compound="left")
        button.place(x=x, y=y, anchor="center")

    exit_button.config(width=button_width, height=button_height, font=("Arial", cached_font_size, "bold"),
                       image=icons["Exit"], compound="left")
    exit_button.place(x=screen_width // 2, y=screen_height * 0.9, anchor="center")

# Initialize Tkinter window
root = tk.Tk()
root.title("Exercise Feedback System")
root.attributes('-fullscreen', True)

# Get screen dimensions
screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()

# Canvas for background gradient
gradient_canvas = tk.Canvas(root, width=screen_width, height=screen_height, highlightthickness=0)
gradient_canvas.pack(fill="both", expand=True)

# Load images
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

# Initialize buttons
exercises = {
    "Bicep Curls": "BicepCurls.py",
    "Posture": "posture.py",
    "Squats": "Squat.py",
    "Wrist Rotation": "WristRotation.py",
    "Lunges": "Lunges.py",
    "Jumping Jacks": "JumpingJacks.py"
}

button_colors = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#66ccff", "#ff66b2"]
buttons = {}

# Create buttons
for index, (exercise, script) in enumerate(exercises.items()):
    img = Image.open(image_files[exercise])
    img = img.resize((50, 50), Image.Resampling.LANCZOS)
    icons[exercise] = ImageTk.PhotoImage(img)

    btn = tk.Button(root, text=exercise, command=lambda s=script: run_model(s), 
                    image=icons[exercise], compound="left", padx=20, pady=10,
                    bg=button_colors[index], font=("Arial", 16, "bold"), width=20)
    buttons[exercise] = btn

# Exit button
exit_img = Image.open(image_files["Exit"])
exit_img = exit_img.resize((50, 50), Image.Resampling.LANCZOS)
icons["Exit"] = ImageTk.PhotoImage(exit_img)

exit_button = tk.Button(root, text="Exit", command=root.quit, image=icons["Exit"], compound="left", 
                        padx=20, pady=10, bg="red", fg="white", font=("Arial", 16, "bold"), width=20)

# Cache previous values
cached_icon_size = -1
cached_font_size = -1

# Bind resize event
root.bind("<Configure>", resize)
root.bind("<Escape>", exit_fullscreen)

# Call resize to set initial layout
root.update_idletasks()
resize()

# Load model at startup
if not load_model():
    print("Warning: Failed to load model at startup. Some features may not work properly.")

root.mainloop()
