import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
model = InceptionV3(weights='imagenet')

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize
        resized_frame = cv2.resize(frame, (299, 299))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_input(np.expand_dims(rgb_frame, axis=0))

        # Predict using the model
        predictions = model.predict(preprocessed)
        decoded = decode_predictions(predictions, top=1)[0]

        # Get the top prediction
        label, confidence = decoded[0][1], decoded[0][2]

        # Filter based on confidence
        if confidence < 0.3:
            label = "Unknown"
            confidence = 0.0

        #(make things cleaner)
        label_mapping = {
            "iPod": "Mobile Phone",
            "cellular telephone": "Mobile Phone",
            "smartphone": "Mobile Phone",
            "laptop": "Laptop",
            "notebook": "Laptop"
        }
        label = label_mapping.get(label, label)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
