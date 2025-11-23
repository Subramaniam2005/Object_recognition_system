import cv2
import os

# Change this folder name depending on what you want to capture
# For pen images use 'images/raw_object'
# For background images use 'images/raw_background'
IMAGE_DIR = 'images/raw_background'

# Create directory if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Open the webcam
cap = cv2.VideoCapture(0)
count = len(os.listdir(IMAGE_DIR))  # Start counting from existing files

print("Press 'c' to capture image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow('Capture Images', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Save the captured frame
        img_name = os.path.join(IMAGE_DIR, f"object_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Image saved: {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
