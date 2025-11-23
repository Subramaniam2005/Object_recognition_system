import cv2
import numpy as np
import tensorflow as tf

# Parameters
IMG_HEIGHT = 64
IMG_WIDTH = 64
MODEL_PATH = 'object_detector.h5'

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Sliding window to scan image
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Open webcam
cap = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    winW, winH = 128, 128  # window size for detection
    best_box = None
    max_confidence = 0.0

    for (x, y, window) in sliding_window(frame, step_size=32, window_size=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Prepare window for model
        roi = cv2.resize(window, (IMG_WIDTH, IMG_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = tf.keras.preprocessing.image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        pred = model.predict(roi, verbose=0)[0][0]
        confidence = 1 - pred  # Confidence that this is your object

        if confidence > 0.90 and confidence > max_confidence:
            max_confidence = confidence
            best_box = (x, y, x + winW, y + winH)

    # Draw bounding box
    if best_box is not None:
        (startX, startY, endX, endY) = best_box
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "Object Detected", (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
