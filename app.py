import cv2
import torch
import pytesseract
import numpy as np
from ultralytics import YOLO

# Path to Tesseract OCR executable (update this according to your installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Class for casting image detection and text extraction
class CastingDetector:
    def __init__(self, model_path="models/best.pt"):
        # Initialize YOLO model with the given model path
        self.model = YOLO(model_path)

    # Detects casting image in a given frame
    def detect_image(self, frame):
        results = self.model(frame)  # Run YOLO model on the frame
        castings = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates for detected castings
        rendered_frame = results[0].plot()  # Visualize detections on the frame
        return castings, rendered_frame

    @staticmethod
    # Extract text from a cropped image of a casting using Tesseract OCR
    def extract_text(cropped_image):
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]  # Apply thresholding for better OCR performance
        text = pytesseract.image_to_string(gray, config="--psm 7").strip()  # Extract text with OCR (page segmentation mode 7)
        return text

# Crop the image based on the given bounding box coordinates
def crop_image(frame, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)  # Extract coordinates from bounding box
    return frame[y_min:y_max, x_min:x_max]  # Crop and return the image of the casting

# Draw text on the frame at the given position
def draw_text(frame, text, position, color=(0, 255, 0), font_scale=1, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Compare the detected casting text with the reference casting
def compare_images(detected_casting, reference_casting):
    return detected_casting == reference_casting  # Return True if the castings match, else False

# Main function that handles the real-time casting image detection and verification
def main():
    detector = CastingDetector()  # Initialize the casting detector
    reference_casting = "CastingA"  # Define the reference casting to verify against

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time casting image verification. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect casting images in the current frame
        castings, rendered_frame = detector.detect_image(frame)

        for casting in castings:
            cropped_image = crop_image(frame, casting)  # Crop the detected casting from the frame
            detected_text = detector.extract_text(cropped_image)  # Extract text from the cropped casting

            # Compare the detected text with the reference casting
            match = compare_images(detected_text, reference_casting)
            result_text = "Match" if match else "No Match"  # Prepare result text based on comparison

            # Draw text and bounding box around the detected casting
            x_min, y_min, x_max, y_max = map(int, casting)
            draw_text(rendered_frame, f"Detected: {detected_text} ({result_text})", (x_min, y_min - 10), color=(255, 0, 0))
            cv2.rectangle(rendered_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the frame with detections and result text
        cv2.imshow("Casting Image Verification", rendered_frame)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the webcam resource
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Entry point of the program
if __name__ == "__main__":
    main()  # Run the main function to start the detection and verification process
