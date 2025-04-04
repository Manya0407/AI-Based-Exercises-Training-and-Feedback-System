# AI-Based Exercises Training and Feedback System

A real-time exercise monitoring and feedback system that uses computer vision and machine learning to track and analyze various exercises, providing instant feedback and performance metrics useing TensorFlow and MoveNet, featuring a responsive  Tkinter GUI 

## Features

- **Real-time Exercise Tracking**: Monitor and analyze exercises in real-time using webcam input
- **Multiple Exercise Types**:
  - Bicep Curls - Tracks both left and right arm curls separately
                - Monitors elbow angles and form
                - Provides feedback on arm synchronization

  - Squats      - Tracks squat depth and form
                - Monitors knee and hip angles
                - Provides feedback on posture and alignment
    
  - Lunges     - Tracks lunge depth and form
               - Monitors knee angles and hip alignment
               - Provides feedback on stance and balance

  - Wrist Rotations    - Tracks clockwise and counterclockwise rotations
                      - Monitors wrist angles and movement
    
  - Posture Analysis  - Monitors overall body posture
                      - Provides real-time feedback on alignment
                      - Tracks posture changes over time
  - Jumping Jacks,    - Tracks repetitions
                      - Monitors arm and leg movement
- **Detailed Feedback**: Get real-time feedback on form and technique
- **Performance Metrics**: Track repetitions, angles, and form consistency
- **Exercise Records**: Save and review exercise history
- **User-Friendly Interface**: Simple GUI with exercise selection and visual feedback

## Screenshots

### Main Interface
![Main Interface](images/main_interface.png)
*The main exercise selection screen with all available exercises*

## Requirements

- Python 3.8 or higher
- Webcam
- Internet connection (for model download)
### Python Packages
- numpy==2.2.3
- opencv_contrib_python==4.9.0.80
- opencv_python==4.11.0.86
- Pillow==11.1.0
- tensorflow==2.18.0
- tensorflow_hub==0.16.1
- tensorflow_intel==2.18.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Based-Exercises-Training-and-Feedback-System.git
cd AI-Based-Exercises-Training-and-Feedback-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python gui.py
```

2. Select an exercise from the GUI interface
3. Position yourself in front of the webcam
4. Follow the on-screen instructions and feedback
5. Press 'q' to quit the exercise
6. Press 'r' to reset counters (where applicable)
7. Press 'ESC' to exit fullscreen mode

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Improvements

- Add more exercise types
- Implement exercise routines and programs
- Add voice feedback
- Support for multiple camera angles
- Mobile app integration
- Social features and challenges

