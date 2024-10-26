from flask import Flask, render_template, Response
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5', compile=False)

def detect(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0 
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    l = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return(l[predicted_class[0]+1])

def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    
    
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    
                    cv2.rectangle(frame, (x_min + 10, y_min + 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)
                    roi = frame[y_min:y_max, x_min:x_max]
                    
                    y_pred = detect(roi)
                    
                    draw_label(frame, y_pred, (50,50), (255,0,0))

                    # mp_drawing.draw_landmarks(
                    #     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    # )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
