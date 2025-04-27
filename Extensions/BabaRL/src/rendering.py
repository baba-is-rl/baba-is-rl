import os

import numpy as np
import pyBaba
import pygame

import sprites
from constants import MAX_MAP_HEIGHT, MAX_MAP_WIDTH
from sprites import BLOCK_SIZE

COLOR_BACKGROUND = pygame.Color(0, 0, 0)

LAYER_PRIORITY = {
    # Floor/Background Tiles
    pyBaba.ObjectType.ICON_TILE: 0,
    pyBaba.ObjectType.ICON_GRASS: 0,
    pyBaba.ObjectType.ICON_WATER: 0,
    pyBaba.ObjectType.ICON_LAVA: 0,
    pyBaba.ObjectType.ICON_EMPTY: -1,
    # Mid-level Objects (Characters, Items)
    pyBaba.ObjectType.ICON_BABA: 10,
    pyBaba.ObjectType.ICON_ROCK: 10,
    pyBaba.ObjectType.ICON_FLAG: 10,
    pyBaba.ObjectType.ICON_WALL: 10,
    pyBaba.ObjectType.ICON_SKULL: 10,
    # Text Objects (Drawn on top)
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
}

DEFAULT_PRIORITY = 15


def get_layer_priority(obj_type):
    return LAYER_PRIORITY.get(obj_type, DEFAULT_PRIORITY)


class Renderer:
    def __init__(self, game, title, sprites_path="../sprites", enable_render=True):
        self.game = game
        self.game_over = False
        self.enable_render = enable_render
        self.screen = None
        self.sprite_loader = None
        self.screen_size = (0, 0)

        if self.enable_render:
            try:
                pygame.init()
                pygame.display.set_caption(title)
                initial_map = game.GetMap()
                map_width = initial_map.GetWidth()
                map_height = initial_map.GetHeight()
                self.screen_size = (map_width * BLOCK_SIZE, map_height * BLOCK_SIZE)

                self.screen = pygame.display.set_mode(
                    (self.screen_size[0], self.screen_size[1]), pygame.DOUBLEBUF
                )
                print(
                    f"Pygame display initialized ({self.screen_size[0]}x{self.screen_size[1]})"
                )

                if not os.path.isdir(
                    os.path.join(sprites_path, "icon")
                ) or not os.path.isdir(os.path.join(sprites_path, "text")):
                    print(
                        f"Warning: Sprite directories not found in {sprites_path}. Check path."
                    )
                    self.sprite_loader = None
                else:
                    print(f"Loading sprites from: {sprites_path}")
                    self.sprite_loader = sprites.SpriteLoader(sprites_path)

            except pygame.error as e:
                print(f"Error initializing Pygame display: {e}")
                self.enable_render = False
                self.screen = None
                try:
                    pygame.quit()
                except Exception as _:
                    pass
            except ImportError:
                print("Error importing sprites module. Ensure sprites.py exists.")
                self.enable_render = False

    def update_game_reference(self, new_game_instance):
        """Updates the game instance the renderer refers to."""
        self.game = new_game_instance

    def update_display_size(self, new_width, new_height):
        """Recreates the Pygame screen with new dimensions if they changed."""
        if not self.enable_render:
            return

        new_screen_size_pixels = (new_width * BLOCK_SIZE, new_height * BLOCK_SIZE)

        if new_screen_size_pixels != self.screen_size:
            print(
                f"Renderer: Resizing display from {self.screen_size} to {new_screen_size_pixels}"
            )
            self.screen_size = new_screen_size_pixels
            try:
                self.screen = pygame.display.set_mode(
                    (self.screen_size[0], self.screen_size[1]),
                    pygame.DOUBLEBUF | pygame.RESIZABLE,
                )
                # self.draw(self.game.GetMap())
                # pygame.display.flip()
            except pygame.error as e:
                print(f"Renderer: Error resizing display: {e}")
                # self.enable_render = False
                # self.screen = None

    def draw_obj(self, map_obj, x_pos, y_pos):
        if not self.enable_render or self.screen is None or self.sprite_loader is None:
            return

        objects = map_obj.At(x_pos, y_pos)
        obj_types = objects.GetTypes()

        sprites_to_draw = []

        for obj_type in obj_types:
            priority = get_layer_priority(obj_type)
            if priority < 0:
                continue

            img_surface = None
            try:
                if pyBaba.IsTextType(obj_type):
                    if obj_type in self.sprite_loader.text_images:
                        img_surface = self.sprite_loader.text_images[obj_type]
                    # else: print(f"Missing text sprite for {obj_type}")
                else:
                    if obj_type in self.sprite_loader.icon_images:
                        img_surface = self.sprite_loader.icon_images[obj_type]
                    # else: print(f"Missing icon sprite for {obj_type}")

                if img_surface:
                    img_rect = img_surface.get_rect()
                    img_rect.topleft = (x_pos * BLOCK_SIZE, y_pos * BLOCK_SIZE)
                    sprites_to_draw.append((priority, img_surface, img_rect))

            except KeyError:
                print(
                    f"Warning: Missing sprite key for object type {obj_type} at ({x_pos}, {y_pos})"
                )
            except AttributeError as e:
                if "NoneType" in str(
                    e
                ) and "'NoneType' object has no attribute 'text_images'" in str(e):
                    pass
                else:
                    print(f"AttributeError during drawing object {obj_type}: {e}")

        sprites_to_draw.sort(key=lambda item: item[0])

        for _, surface, rect in sprites_to_draw:
            try:
                self.screen.blit(surface, rect)
            except pygame.error as blit_error:
                print(f"Error blitting sprite at {rect.topleft}: {blit_error}")
                self.game_over = True
                break

    def draw(self, map_obj):
        if not self.enable_render or self.screen is None or self.game_over:
            return
        try:
            current_map_height = map_obj.GetHeight()
            current_map_width = map_obj.GetWidth()

            self.screen.fill(COLOR_BACKGROUND)
            for y_pos in range(current_map_height):
                for x_pos in range(current_map_width):
                    self.draw_obj(map_obj, x_pos, y_pos)
        except pygame.error as e:
            print(f"Error during drawing background/looping: {e}")
            self.game_over = True
        except AttributeError:
            print("Error: Invalid map object passed to draw.")
            self.game_over = True

    def render(self, map_obj, mode="human"):
        if not self.enable_render or self.screen is None:
            if mode == "rgb_array":
                h = MAX_MAP_HEIGHT * BLOCK_SIZE
                w = MAX_MAP_WIDTH * BLOCK_SIZE
                return np.zeros((h, w, 3), dtype=np.uint8)
            return None

        self.process_event()

        if self.game_over:
            if mode == "rgb_array":
                h = MAX_MAP_HEIGHT * BLOCK_SIZE
                w = MAX_MAP_WIDTH * BLOCK_SIZE
                return np.zeros((h, w, 3), dtype=np.uint8)
            return None

        self.draw(map_obj)

        if mode == "human":
            try:
                pygame.display.flip()
            except pygame.error as e:
                print(f"Error during pygame.display.flip(): {e}")
                self.game_over = True
                return None
        elif mode == "rgb_array":
            try:
                return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
            except pygame.error as e:
                print(f"Error getting rgb_array: {e}")
                self.game_over = True
                h = MAX_MAP_HEIGHT * BLOCK_SIZE
                w = MAX_MAP_WIDTH * BLOCK_SIZE
                return np.zeros((h, w, 3), dtype=np.uint8)

        return None 

    def process_event(self):
        if not self.enable_render:
            return
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event received.")
                    self.game_over = True
                    self.quit_game()
        except pygame.error as e:
            if "video system not initialized" not in str(e):
                print(f"Pygame error during event processing: {e}")
            self.game_over = True

    def quit_game(self):
        if self.enable_render:
            print("Attempting Pygame quit...")
            try:
                pygame.display.quit()
            except pygame.error as e:
                if "display Surface quit" not in str(
                    e
                ) and "video system not initialized" not in str(e):
                    print(f"Error during pygame.display.quit(): {e}")
            try:
                pygame.quit()
                print("Pygame quit successfully.")
            except pygame.error as e:
                if "video system not initialized" not in str(e):
                    print(f"Error during pygame.quit(): {e}")
