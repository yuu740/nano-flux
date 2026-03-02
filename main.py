import cv2
import pygame
import sys
from core.particle import Particle
from core.shape_manager import ShapeManager
from engines.vision import VisionEngine

WIDTH, HEIGHT = 1200, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Nano Flux - Particle Bender")
clock = pygame.time.Clock()

sm = ShapeManager()
vision = VisionEngine()
cap = cv2.VideoCapture(0)

particles = [Particle(900, 300) for _ in range(1000)]
welcome_triggered = False
current_name = ""

while True:
    screen.fill((10, 10, 10))
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    hand_landmarks, gesture, recognized_name = vision.process_frame(frame)

    targets = []
    if not welcome_triggered and recognized_name != "Unknown":
        current_name = recognized_name.upper()
        targets = sm.get_text_targets(f"WELCOME {current_name}")
        welcome_triggered = True
    elif gesture == "PEACE":
        targets = sm.get_text_targets("PEACE ✌")
    elif gesture == "LOVE":
        targets = sm.get_text_targets("LOVE ❤")
    else:
        targets = sm.get_text_targets("NANO FLUX")

    for i, p in enumerate(particles):
        t = targets[i % len(targets)]
        p.update_target(t[0], t[1])
        
        h_pos = None
        if hand_landmarks:
            h_pos = (hand_landmarks[9].x * 600 + 600, hand_landmarks[9].y * 600)
            
        p.apply_behaviors(h_pos, gesture)
        p.update()
        p.draw(screen)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    cam_surface = pygame.transform.scale(cam_surface, (600, 600))
    screen.blit(cam_surface, (0, 0))
    
    font = pygame.font.SysFont("Arial", 18)
    info_text = f"User: {recognized_name} | Gesture: {gesture}"
    txt_surf = font.render(info_text, True, (0, 255, 0))
    screen.blit(txt_surf, (10, 10))

    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
    clock.tick(60)