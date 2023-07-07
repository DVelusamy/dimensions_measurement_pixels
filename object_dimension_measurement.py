import cv2
import numpy as np

def measure_object_dimensions():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform thresholding
        _, thresholded_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        # Find contours of objects in the frame
        contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (complete object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding rectangle on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the dimensions on the frame
        cv2.putText(frame, f"Width: {w} pixels", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Height: {h} pixels", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Object Dimension Measurement", frame)

        # Check for the 'q' key to quit the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or Esc key
            break

    cap.release()
    cv2.destroyAllWindows()

measure_object_dimensions()
