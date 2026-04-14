# AI GAZE - Predictive Eye Tracker Application

## Overview
The AI GAZE application uses computer vision to predict user gaze direction and provides various functionalities.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- TensorFlow/Keras (for any ML models)

## Installation
To install the required packages, run:
```bash
pip install opencv-python numpy tensorflow keras
```

## Code
```python
import cv2
import numpy as np

class EyeTracker:
    def __init__(self):
        # Initialize the video capture object
        self.capture = cv2.VideoCapture(0)  # Use 0 for the primary camera

    def predict_gaze(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            # [Insert gaze prediction logic here]
            # For example, using a pre-trained model

            # Display the resulting frame
            cv2.imshow('Eye Tracker', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = EyeTracker()
    tracker.predict_gaze()
```

## Instructions to Run
To run the application, execute:
```bash
python app.py
```