import pyBaba
import pygame

BLOCK_SIZE = 48


class SpriteLoader:
    def __init__(self, path: str):
        self.icon_names = {
            pyBaba.ObjectType.ICON_BABA: "BABA",
            pyBaba.ObjectType.ICON_FLAG: "FLAG",
            pyBaba.ObjectType.ICON_WALL: "WALL",
            pyBaba.ObjectType.ICON_ROCK: "ROCK",
            pyBaba.ObjectType.ICON_TILE: "TILE",
            pyBaba.ObjectType.ICON_WATER: "WATER",
            pyBaba.ObjectType.ICON_GRASS: "GRASS",
            pyBaba.ObjectType.ICON_TILE: "TILE",
            pyBaba.ObjectType.ICON_LAVA: "LAVA",
        }

        self.icon_images = {}
        for i in self.icon_names:
            self.icon_images[i] = pygame.transform.scale(
                pygame.image.load(f"{path}/icon/{self.icon_names[i]}.gif"),
                (BLOCK_SIZE, BLOCK_SIZE),
            )

        self.text_names = {
            pyBaba.ObjectType.BABA: "BABA",
            pyBaba.ObjectType.IS: "IS",
            pyBaba.ObjectType.YOU: "YOU",
            pyBaba.ObjectType.FLAG: "FLAG",
            pyBaba.ObjectType.WIN: "WIN",
            pyBaba.ObjectType.WALL: "WALL",
            pyBaba.ObjectType.STOP: "STOP",
            pyBaba.ObjectType.ROCK: "ROCK",
            pyBaba.ObjectType.PUSH: "PUSH",
            pyBaba.ObjectType.WATER: "WATER",
            pyBaba.ObjectType.SINK: "SINK",
            pyBaba.ObjectType.LAVA: "LAVA",
            pyBaba.ObjectType.MELT: "MELT",
            pyBaba.ObjectType.HOT: "HOT",
        }

        self.text_images = {}
        for i in self.text_names:
            self.text_images[i] = pygame.transform.scale(
                pygame.image.load(f"{path}/text/{self.text_names[i]}.gif"),
                (BLOCK_SIZE, BLOCK_SIZE),
            )
