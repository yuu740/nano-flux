import cv2
import mediapipe as mp
import numpy as np
import os

class VisionEngine:
    def __init__(self, assets_path="assets/"):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        
        self.known_face_signatures = {}
        self.load_assets(assets_path)

    def get_face_signature(self, landmarks):
        points = np.array([[l.x, l.y] for l in landmarks.landmark])
        center = points[1] 
        signature = points - center
        return signature

    def load_assets(self, path):
        if not os.path.exists(path): return
        for file in os.listdir(path):
            if file.endswith((".png", ".jpg")):
                img = cv2.imread(os.path.join(path, file))
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = self.face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    name = os.path.splitext(file)[0]
                    self.known_face_signatures[name] = self.get_face_signature(res.multi_face_landmarks[0])

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_res = self.hands.process(rgb_frame)
        hand_data, gesture = None, "IDLE"
        if hand_res.multi_hand_landmarks:
            hand_data = hand_res.multi_hand_landmarks[0].landmark
            if hand_data[8].y < hand_data[6].y and hand_data[12].y < hand_data[10].y: gesture = "PEACE"
            elif np.linalg.norm(np.array([hand_data[4].x, hand_data[4].y]) - np.array([hand_data[8].x, hand_data[8].y])) < 0.05: gesture = "LOVE"

        face_res = self.face_mesh.process(rgb_frame)
        face_name = "Unknown"
        if face_res.multi_face_landmarks:
            current_sig = self.get_face_signature(face_res.multi_face_landmarks[0])
            for name, known_sig in self.known_face_signatures.items():
                diff = np.linalg.norm(current_sig - known_sig)
                if diff < 0.2: 
                    face_name = name
                    break
        
        return hand_data, gesture, face_name