import pygame

class ShapeManager:
    def __init__(self, font_size=80):
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)

    def get_text_targets(self, text, offset_x=900, offset_y=300):
        surface = self.font.render(text, True, (255, 255, 255))
        w, h = surface.get_size()
        pixels = pygame.PixelArray(surface)
        
        targets = []
        for y in range(0, h, 4): 
            for x in range(0, w, 4):
                if pixels[x, y] > 0:
                    targets.append((x + offset_x - w//2, y + offset_y - h//2))
        return targets