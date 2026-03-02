import cv2
import mediapipe as mp
import pygame
import random
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Particle:
    def __init__(self):
        self.pos = np.array([random.uniform(0, WIDTH), random.uniform(0, HEIGHT)])
        self.vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.acc = np.array([0.0, 0.0])
        self.max_speed = 5

    def apply_force(self, force):
        self.acc += force

    def update(self, target=None, mode="FLOAT"):
        if mode == "ATTRACT" and target is not None:
            dir_vec = target - self.pos
            dist = np.linalg.norm(dir_vec)
            if dist > 0:
                dir_vec = (dir_vec / dist) * 0.5
                self.apply_force(dir_vec)
        
        elif mode == "EXPLODE" and target is not None:
            dir_vec = self.pos - target
            dist = np.linalg.norm(dir_vec)
            if dist < 100:
                dir_vec = (dir_vec / (dist + 1)) * 2
                self.apply_force(dir_vec)

        self.vel += self.acc
        self.vel = np.clip(self.vel, -self.max_speed, self.max_speed)
        self.pos += self.vel
        self.acc *= 0 
        
        if self.pos[0] <= 0 or self.pos[0] >= WIDTH: self.vel[0] *= -1
        if self.pos[1] <= 0 or self.pos[1] >= HEIGHT: self.vel[1] *= -1

    def draw(self):
        pygame.draw.circle(screen, (0, 255, 255), self.pos.astype(int), 2)

particles = [Particle() for _ in range(500)]
cap = cv2.VideoCapture(0)

running = True
mode = "FLOAT"

while running:
    screen.fill((0, 0, 0))
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    target_pos = None
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        cx = int(hand_landmarks[9].x * WIDTH)
        cy = int(hand_landmarks[9].y * HEIGHT)
        target_pos = np.array([cx, cy])
        
        dist_pinch = np.linalg.norm(np.array([hand_landmarks[4].x, hand_landmarks[4].y]) - 
                                   np.array([hand_landmarks[8].x, hand_landmarks[8].y]))
        
        if dist_pinch < 0.05: mode = "ATTRACT"
        else: mode = "EXPLODE"

    for p in particles:
        p.update(target_pos, mode)
        p.draw()

    pygame.display.flip()
    clock.tick(60)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

cap.release()
pygame.quit()