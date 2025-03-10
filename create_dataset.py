"""
Code to create a dataset based on photos made with the webCam
"""
import cv2
import os

# Folder where images will be saved
folder = "dataset"

# Initialize the camera (0 for the default camera)
cap = cv2.VideoCapture(0)

# Counter for naming saved image files
counter = 1

# Create a window with instructions
cv2.namedWindow("Press 'c' to capture - 'q' to exit")

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if the frame is not captured properly
    
    # Display the captured frame in the window
    cv2.imshow("Press 'c' to capture - 'q' to exit", frame)

    # Capture the pressed key
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # If 'c' is pressed, capture the image
        filename = os.path.join(folder, f"{counter}.png")  # Generate the full file path
        cv2.imwrite(filename, frame)  # Save the image to the specified folder
        print(f"Image saved as {filename}")
        counter += 1  # Increment the file counter for the next image

    elif key == ord('q'):  # If 'q' is pressed, exit the loop
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
