import cv2
import numpy as np

# Initialize the camera (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Initialize variables for motion detection
previous_frame = None

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve accuracy
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the previous frame if it's the first frame
    if previous_frame is None:
        previous_frame = gray
        continue

    # Compute the absolute difference between the current and previous frame
    frame_delta = cv2.absdiff(previous_frame, gray)

    # Apply a threshold to the frame delta
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours (noise)
        if cv2.contourArea(contour) < 1000:
            continue

        # Draw a bounding box around the detected motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the original frame with motion detection overlay
    cv2.imshow("Motion Detection", frame)

    # Update the previous frame
    previous_frame = gray

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
