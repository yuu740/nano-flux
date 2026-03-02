import numpy as np
import pygame
import random

class Particle:
    def __init__(self, x, y):
        self.pos = np.array([random.uniform(600, 1200), random.uniform(0, 600)], dtype=float)
        self.target = np.array([x, y], dtype=float)
        self.vel = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
        self.acc = np.array([0.0, 0.0], dtype=float)
        self.max_speed = 10
        self.max_force = 0.5

    def update_target(self, new_x, new_y):
        self.target = np.array([new_x, new_y], dtype=float)

    def apply_behaviors(self, hand_pos, mode):
        arrive = self.arrive(self.target)
        self.acc += arrive
        
        if hand_pos is not None:
            flee = self.flee(hand_pos)
            self.acc += flee * 2

    def arrive(self, target):
        desired = target - self.pos
        dist = np.linalg.norm(desired)
        speed = self.max_speed
        if dist < 100:
            speed = np.interp(dist, [0, 100], [0, self.max_speed])
        
        desired = (desired / (dist + 1e-6)) * speed
        return np.clip(desired - self.vel, -self.max_force, self.max_force)

    def flee(self, target):
        desired = self.pos - target
        dist = np.linalg.norm(desired)
        if dist < 80:
            desired = (desired / (dist + 1e-6)) * self.max_speed
            return np.clip(desired - self.vel, -self.max_force, self.max_force)
        return np.array([0, 0], dtype=float)

    def update(self):
        self.vel += self.acc
        self.pos += self.vel
        self.acc *= 0

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 255), self.pos.astype(int), 2)