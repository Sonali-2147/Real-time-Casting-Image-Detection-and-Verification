# Real-time-Casting-Image-Detection-and-Verification

This project implements real-time casting image detection and verification using YOLOv8 and Tesseract OCR. The system uses a webcam to capture frames, detects casting images in the video feed, and compares the detected castings with a predefined reference casting. It also extracts text from the detected castings using Tesseract OCR for verification.

## Features

- **Real-time detection**: Detects casting images in live video feed using a YOLOv8 model.
- **Text extraction**: Extracts text from detected casting images using Tesseract OCR.
- **Casting comparison**: Compares the detected casting text with a reference casting for matching.
- **Real-time results**: Displays the detected text and match status (Match / No Match) on the video feed.

## Prerequisites

Before running the project, make sure you have the following dependencies installed:

- Python 3.6+
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/yolov8) for object detection
- [OpenCV](https://opencv.org/) for image processing and video capture
- [Pytesseract](https://pypi.org/project/pytesseract/) for text extraction from images

Additionally, you will need to have **Tesseract OCR** installed on your system.

### Install Required Packages

To install the necessary Python packages, you can use `pip`:

```bash
pip install opencv-python pytesseract ultralytics torch numpy
```

### Install Tesseract OCR

Download and install Tesseract from the official repository:

- Windows: [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

## Setup

1. **Download the pre-trained YOLOv8 model**:
   - Make sure to download a custom-trained YOLO model for casting image detection. Save it as `best.pt` in the `models/` directory.

2. **Configure Tesseract**:
   - Set the correct path to your Tesseract OCR executable in the `pytesseract.pytesseract.tesseract_cmd` variable. The default path in the code is for a Windows installation. Update it accordingly for your operating system.

3. **Video Capture**:
   - The script uses your system's webcam for real-time video capture. Ensure the webcam is working correctly.

## Running the Application

To run the application, simply execute the Python script:

```bash
python app.py
```

The webcam feed will appear, and the system will begin detecting and verifying casting images. The detected casting's text will be displayed along with the comparison result ("Match" or "No Match"). Press the 'q' key to exit the application.

### Example Output:

- **Detected**: CastingA (Match)
- **Detected**: CastingB (No Match)

## Code Overview

### `CastingDetector` Class
- **`__init__(self, model_path)`**: Initializes the YOLOv8 model with the provided model path (`best.pt`).
- **`detect_image(self, frame)`**: Detects casting images in the provided image frame and returns their bounding box coordinates and the visualized frame.
- **`extract_text(cropped_image)`**: Extracts text from the cropped casting image using Tesseract OCR.

### Helper Functions
- **`crop_image(frame, bbox)`**: Crops the frame based on the given bounding box coordinates.
- **`draw_text(frame, text, position)`**: Draws the result text on the frame at the specified position.
- **`compare_images(detected_casting, reference_casting)`**: Compares the detected casting text with the reference casting for a match.

### `main()` Function
- Opens the webcam, captures frames, detects casting images, extracts text, and compares with the reference casting in real-time. Displays the results in a video window and allows the user to quit by pressing 'q'.

## Contact
For any queries or feedback, please email me at kadamsonali2147@gmail.com.
