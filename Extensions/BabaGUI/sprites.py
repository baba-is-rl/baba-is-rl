import os

import pyBaba
import pygame

BLOCK_SIZE = 24

LAYER_PRIORITY = {
    # Background tiles (Layer 0)
    pyBaba.ObjectType.ICON_TILE: 0,
    pyBaba.ObjectType.ICON_GRASS: 0,
    pyBaba.ObjectType.ICON_WATER: 0,
    pyBaba.ObjectType.ICON_LAVA: 0,
    # Mid-level objects (Layer 10)
    pyBaba.ObjectType.ICON_WALL: 10,
    pyBaba.ObjectType.ICON_ROCK: 10,
    pyBaba.ObjectType.ICON_FLAG: 10,
    pyBaba.ObjectType.ICON_SKULL: 10,
    pyBaba.ObjectType.ICON_FLOWER: 10,
    pyBaba.ObjectType.ICON_BABA: 11,
    # Text objects (Layer 20)
    pyBaba.ObjectType.BABA: 20,
    pyBaba.ObjectType.IS: 20,
    pyBaba.ObjectType.YOU: 20,
    pyBaba.ObjectType.FLAG: 20,
    pyBaba.ObjectType.WIN: 20,
    pyBaba.ObjectType.WALL: 20,
    pyBaba.ObjectType.STOP: 20,
    pyBaba.ObjectType.ROCK: 20,
    pyBaba.ObjectType.PUSH: 20,
    pyBaba.ObjectType.WATER: 20,
    pyBaba.ObjectType.SINK: 20,
    pyBaba.ObjectType.LAVA: 20,
    pyBaba.ObjectType.MELT: 20,
    pyBaba.ObjectType.HOT: 20,
    pyBaba.ObjectType.SKULL: 20,
    pyBaba.ObjectType.DEFEAT: 20,
    pyBaba.ObjectType.FLOWER: 20,
    pyBaba.ObjectType.ICON_EMPTY: -1,  # Don't draw empty explicitly
}

DEFAULT_PRIORITY = 15


def get_layer_priority(obj_type):
    return LAYER_PRIORITY.get(obj_type, DEFAULT_PRIORITY)


class SpriteLoader:
    def __init__(self, path: str):
        self.base_path = path
        self.icon_images = {}
        self.text_images = {}
        self.result_images = {}
        self.result_images_loaded = False

        icon_names = {
            pyBaba.ObjectType.ICON_BABA: "BABA",
            pyBaba.ObjectType.ICON_FLAG: "FLAG",
            pyBaba.ObjectType.ICON_WALL: "WALL",
            pyBaba.ObjectType.ICON_ROCK: "ROCK",
            pyBaba.ObjectType.ICON_TILE: "TILE",
            pyBaba.ObjectType.ICON_WATER: "WATER",
            pyBaba.ObjectType.ICON_GRASS: "GRASS",
            pyBaba.ObjectType.ICON_LAVA: "LAVA",
            pyBaba.ObjectType.ICON_SKULL: "SKULL",
            pyBaba.ObjectType.ICON_FLOWER: "FLOWER",
        }
        text_names = {
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
            pyBaba.ObjectType.SKULL: "SKULL",
            pyBaba.ObjectType.DEFEAT: "DEFEAT",
            pyBaba.ObjectType.FLOWER: "FLOWER",
        }

        icon_dir = os.path.join(self.base_path, "icon")
        if not os.path.isdir(icon_dir):
            print(f"Warning: Icon directory not found: {icon_dir}")
        else:
            for obj_type, name in icon_names.items():
                self.icon_images[obj_type] = self._load_and_scale(icon_dir, name)

        text_dir = os.path.join(self.base_path, "text")
        if not os.path.isdir(text_dir):
            print(f"Warning: Text directory not found: {text_dir}")
        else:
            for obj_type, name in text_names.items():
                self.text_images[obj_type] = self._load_and_scale(text_dir, name)

        try:
            won_img_path = os.path.join(self.base_path, "won.png")
            lost_img_path = os.path.join(self.base_path, "lost.png")
            self.result_images["won"] = pygame.image.load(won_img_path)
            self.result_images["lost"] = pygame.image.load(lost_img_path)
            self.result_images_loaded = True
            print("Loaded result images (won.png, lost.png)")
        except pygame.error as e:
            print(
                f"Warning: Could not load result images (won.png/lost.png) from {self.base_path}: {e}"
            )
            self.result_images_loaded = False

    def _load_and_scale(self, directory, name):
        if name is None:
            return None
        file_path = os.path.join(directory, f"{name}.gif")
        try:
            image = pygame.image.load(file_path).convert_alpha()
            return pygame.transform.scale(image, (BLOCK_SIZE, BLOCK_SIZE))
        except pygame.error as e:
            print(f"Warning: Failed to load or scale sprite '{file_path}': {e}")
            return None


class ResultImage(pygame.sprite.Sprite):
    def __init__(self, sprite_loader):
        super().__init__()
        self.sprite_loader = sprite_loader
        self.image = pygame.Surface([0, 0])
        self.rect = self.image.get_rect()
        self.current_state = None

    def update(self, status, screen_size):
        if not self.sprite_loader or not self.sprite_loader.result_images_loaded:
            return

        state_changed = status != self.current_state
        self.current_state = status

        if state_changed:
            img_key = None
            if status == pyBaba.PlayState.WON:
                img_key = "won"
            elif status == pyBaba.PlayState.LOST:
                img_key = "lost"

            if img_key and img_key in self.sprite_loader.result_images:
                original_image = self.sprite_loader.result_images[img_key]
                target_size = min(screen_size[0], screen_size[1]) // 2
                try:
                    self.image = pygame.transform.smoothscale(
                        original_image, (target_size, target_size)
                    )
                except ValueError:
                    self.image = pygame.transform.scale(
                        original_image, (target_size, target_size)
                    )

                self.rect = self.image.get_rect(
                    center=(screen_size[0] // 2, screen_size[1] // 2)
                )
            else:
                self.image = pygame.Surface([0, 0])
                self.rect = self.image.get_rect()
