import cv2
import numpy as np
import dlib

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread

def start_alarm(sound):
    """Play the alarm sound"""
    playsound(r"C:\Users\Abish\Downloads\80s-alarm-clock-sound\80s-alarm-clock-sound.mp3")

face_cascade = cv2.CascadeClassifier(r"C:\Users\Abish\Downloads\data\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(r"C:\Users\Abish\OneDrive\Desktop\Final year project\shape_predictor_68_face_landmarks.dat") #dlib predictor
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

model = load_model("drowiness_cnn.h5")

count = 0
alarm_on = False
alarm_sound = r"C:\Users\Abish\Downloads\80s-alarm-clock-sound\80s-alarm-clock-sound.mp3"
eye_closed = False
mouth_open = False

def is_yawning(landmarks):
    """Checks if the mouth is open based on facial landmarks."""
    top_lip = landmarks[62]  # Landmark for the top lip center
    bottom_lip = landmarks[66] # Landmark for the bottom lip center
    lip_distance = abs(top_lip[1] - bottom_lip[1]) #calculate the distance between the lips
    if lip_distance > 10: #adjust this threshold as needed
        return True
    else:
        return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    height, width, channels = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray) #dlib face detection

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        left_eye_region = landmarks[36:42]
        right_eye_region = landmarks[42:48]

        left_eye_center = np.mean(left_eye_region, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_region, axis=0).astype(int)

        left_eye_roi = frame[left_eye_center[1] - 30:left_eye_center[1] + 30, left_eye_center[0] - 30:left_eye_center[0] + 30]
        right_eye_roi = frame[right_eye_center[1] - 30:right_eye_center[1] + 30, right_eye_center[0] - 30:right_eye_center[0] + 30]

        if left_eye_roi.shape[0] > 0 and left_eye_roi.shape[1] > 0 and right_eye_roi.shape[0] > 0 and right_eye_roi.shape[1] > 0 : #added to prevent errors.
            left_eye_roi = cv2.resize(left_eye_roi, (145,145))
            right_eye_roi = cv2.resize(right_eye_roi, (145,145))

            left_eye_roi = left_eye_roi.astype('float') / 255.0
            right_eye_roi = right_eye_roi.astype('float') / 255.0

            left_eye_roi = img_to_array(left_eye_roi)
            right_eye_roi = img_to_array(right_eye_roi)

            left_eye_roi = np.expand_dims(left_eye_roi, axis=0)
            right_eye_roi = np.expand_dims(right_eye_roi, axis=0)

            pred1 = model.predict(left_eye_roi)
            pred2 = model.predict(right_eye_roi)

            status1 = np.argmax(pred1)
            status2 = np.argmax(pred2)

            eye_closed = (status1 == 2 and status2 == 2) #closed eyes

        mouth_open = is_yawning(landmarks)

        if eye_closed or mouth_open:
            count += 1
            if eye_closed and mouth_open:
                cv2.putText(frame, "Eyes Closed and Yawning, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            elif eye_closed:
                cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.putText(frame, "Yawning, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            if count >= 5:
                cv2.putText(frame, "Drowsiness Alert!!!", (100, height - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound,))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Eyes Open and no yawn", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            count = 0
            alarm_on = False

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
