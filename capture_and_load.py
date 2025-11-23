import cv2
import tensorflow as tf
import os

MODEL_PATH = 'object_detector.h5'  # model file to load

# Check if model exists
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print('Model loaded successfully.')
else:
    print('Model file not found. Please ensure object_detector.h5 exists.')

# Initialize webcam
cap = cv2.VideoCapture(0)
print('Press c to capture image, q to quit.')

capture_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Unable to access camera.')
        break

    cv2.imshow('Webcam Feed', frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        # capture and save image
        img_name = f'captured_{capture_counter}.jpg'
        cv2.imwrite(img_name, frame)
        print(f'Image captured and saved as {img_name}')
        capture_counter += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
