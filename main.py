import os
import sys
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
import numpy as np
from scipy.spatial import distance as dist
import time

# Add Silent-Face-Anti-Spoofing folder to sys.path (keep your existing path)
sys.path.append(r"C:\Users\lavan\OneDrive\Documents\Desktop\opencv\face-attendance-system\Silent-Face-Anti-Spoofing")
from my_test import test  # anti-spoofing function


# --------- Parameters you can tune ----------
BLINK_WINDOW_SECONDS = 5      # blink must happen within this many seconds of login attempt
MIN_FACE_FRACTION = 0.04      # minimum fraction of frame area that the primary face must occupy (4%)
ANTI_SPOOF_THRESHOLD = 0.9    # threshold for anti-spoof model
BLINK_EAR_THRESH = 0.20       # EAR threshold to count as blink
# --------------------------------------------


# ------------- Blink detection (EAR method) ----------------
def eye_aspect_ratio(eye):
    # compute EAR (expects list of (x,y) pairs)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear


def detect_blink(face_landmarks, blink_thresh=BLINK_EAR_THRESH):
    # face_landmarks is dict from face_recognition.face_landmarks
    leftEye = face_landmarks.get("left_eye", [])
    rightEye = face_landmarks.get("right_eye", [])
    if len(leftEye) < 6 or len(rightEye) < 6:
        return False
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear < blink_thresh


# -----------------------------------------------------------

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        # Buttons
        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'register new user', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        # Webcam label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        # DB and log
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = './log.txt'

        # Blink control
        self.blink_detected = False
        self.last_blink_time = None

    # Webcam initialization
    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open webcam. Check camera index or drivers.")
        self._label = label
        self.process_webcam()

    # Process webcam and draw per-face boxes
    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            self._label.after(50, self.process_webcam)
            return

        self.most_recent_capture_arr = frame.copy()
        display_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        # Detect all faces (face_recognition uses HOG/NN)
        face_locations = face_recognition.face_locations(display_frame)
        face_landmarks_list = face_recognition.face_landmarks(display_frame)

        # If there are faces, compute areas and pick the largest (primary) face
        primary_idx = None
        if face_locations:
            areas = []
            for (top, right, bottom, left) in face_locations:
                w = max(1, right - left)
                h = max(1, bottom - top)
                areas.append(w * h)
            primary_idx = int(np.argmax(areas))

        # Draw boxes for all faces (visual feedback) but only process primary face for liveness/recognition
        for i, (top, right, bottom, left) in enumerate(face_locations):
            color = (200, 200, 200)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 1)
            # label primary visually
            if i == primary_idx:
                cv2.putText(display_frame, "Primary", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Process primary face only
        if primary_idx is not None:
            top, right, bottom, left = face_locations[primary_idx]
            landmarks = face_landmarks_list[primary_idx] if primary_idx < len(face_landmarks_list) else None
            # compute primary face area fraction
            primary_area = (right - left) * (bottom - top)
            primary_fraction = primary_area / float(frame_area)

            # If primary face is very small -> likely a card/photo; mark as fake for drawing/feedback
            if primary_fraction < MIN_FACE_FRACTION:
                # Draw big red label near primary bbox
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(display_frame, "Suspected Photo/Card", (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Do NOT set blink_detected or allow recognition for this face
            else:
                # Extract face crop for anti-spoofing model (use primary face)
                face_img = display_frame[top:bottom, left:right].copy()
                try:
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (224, 224))
                    face_resized = face_resized.astype('float32') / 255.0

                    pred_score = test(
                        image=face_resized,
                        model_dir=r"C:\Users\lavan\OneDrive\Documents\Desktop\opencv\face-attendance-system\Silent-Face-Anti-Spoofing\resources\anti_spoof_models",
                        device_id=0
                    )
                    label_pred = 1 if pred_score >= ANTI_SPOOF_THRESHOLD else 0  # 1=real,0=fake

                except Exception as e:
                    label_pred = 0
                    print("Error in test():", e)

                # Blink detection using landmarks (face_recognition returns pixel coords)
                if landmarks and detect_blink(landmarks):
                    self.blink_detected = True
                    self.last_blink_time = time.time()
                    cv2.putText(display_frame, "Blink detected", (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Draw box + label for primary face based on anti-spoof result
                color = (0, 255, 0) if label_pred == 1 else (0, 0, 255)
                text = "Real" if label_pred == 1 else "Fake"
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display in Tkinter
        img_ = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    # --- Login, Logout, Register ---
    def login(self):
        self._attempt_login("in")

    def logout(self):
        self._attempt_login("out")

    def _attempt_login(self, action_type):
        # first ensure we have a current primary face
        frame = self.most_recent_capture_arr
        if frame is None:
            util.msg_box('Error', 'No frame captured.')
            return

        # re-run detection (quick) to get primary face and landmarks
        face_locations = face_recognition.face_locations(frame)
        face_landmarks_list = face_recognition.face_landmarks(frame)

        if not face_locations:
            util.msg_box('No face', 'No face detected. Try again.')
            return

        # pick largest face
        areas = [ (r - l) * (b - t) for (t, r, b, l) in face_locations ]
        primary_idx = int(np.argmax(areas))
        top, right, bottom, left = face_locations[primary_idx]
        primary_area = areas[primary_idx]
        frame_area = frame.shape[0] * frame.shape[1]
        primary_fraction = primary_area / float(frame_area)

        # if primary too small → reject immediately as photo/card
        if primary_fraction < MIN_FACE_FRACTION:
            util.msg_box('Fake / Invalid', 'Detected face is too small.')
            # reset blink
            self.blink_detected = False
            self.last_blink_time = None
            return

        # anti-spoof on primary face
        face_img = frame[top:bottom, left:right].copy()
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_resized = face_resized.astype('float32') / 255.0

            pred_score = test(
                image=face_resized,
                model_dir=r"C:\Users\lavan\OneDrive\Documents\Desktop\opencv\face-attendance-system\Silent-Face-Anti-Spoofing\resources\anti_spoof_models",
                device_id=0
            )
            label = 1 if pred_score >= ANTI_SPOOF_THRESHOLD else 0
        except Exception as e:
            label = 0
            print("Error in test():", e)

        # blink must have happened recently
        blink_ok = (self.blink_detected and self.last_blink_time and (time.time() - self.last_blink_time < BLINK_WINDOW_SECONDS))

        if label == 1 and blink_ok:
            name = util.recognize(frame, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                if action_type == "in":
                    util.msg_box('Welcome back !', f'Welcome, {name}.')
                else:
                    util.msg_box('Hasta la vista !', f'Goodbye, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},{action_type}\n')
        else:
            # If anti-spoof model says real but no blink -> still reject (we want blink every time)
            util.msg_box('Fake / Invalid', 'Liveness failed (no blink detected recently or spoofed).')

        # Reset blink for next attempt
        self.blink_detected = False
        self.last_blink_time = None

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, \ninput username:'
        )
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
        file_path = os.path.join(self.db_dir, f'{name}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)
        util.msg_box('Success!', 'User was registered successfully !')
        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
