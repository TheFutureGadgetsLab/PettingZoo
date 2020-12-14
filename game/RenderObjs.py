import pygame
from pygame import Vector2
import game.defs as pz

class Text:
    def __init__(self, surface, loc):
        self.text = []
        self.surface = surface

        self.position = (0, 0)

        self.loc = loc
        
        self.font_size = 24
        self.font = pygame.font.SysFont("arial", self.font_size, False, False)

    def update(self, text):
        self.text = []
        split = []
        if type(text) == str:
            split = text.split("\n")
        else:
            split = text

        for line in split:
            self.text.append(self.font.render(line, True, (0, 0, 0)))
    
        self.getPos(self.loc)

        
    def getPos(self, pos):
        rect = self.surface.get_rect()
        if pos == "topLeft":
            self.position = rect.topleft
        if pos == "topRight":
            mw = max([t.get_rect().width for t in self.text])
            x = rect.topright[0] - mw - 5
            y = rect.topright[1]
            self.position = (x, y)
        if pos == "topCenter":
            self.position = (Vector2(rect.topleft) + Vector2(rect.topright)) / 2.0

    def draw(self):
        for surf in range(len(self.text)):
            self.surface.blit(self.text[surf], (self.position[0], self.position[1] + (surf * self.font_size)))

class Tile(pygame.sprite.Sprite):
    def __init__(self, image, pos=None) -> None:
        super().__init__()

        self.image = image
        self.rect = self.image.get_rect()
        if pos != None:
            self.rect.x = pos.x
            self.rect.y = pos.y
    
    def update(self, pos):
        self.rect.x = pos.x
        self.rect.y = pos.y

    def draw(self, screen):
        screen.blit(self.image, self.rect)

class Player(Tile):
    def __init__(self, image, pos=None) -> None:
        super().__init__(image, pos=pos)

    def update(self, pos):
        self.rect.x = pos.x - pz.TILE_SIZE // 2
        self.rect.y = pos.y - pz.TILE_SIZE // 2