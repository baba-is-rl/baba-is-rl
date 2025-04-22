import pyBaba
import pygame

import sprites

BLOCK_SIZE = 48
COLOR_BACKGROUND = pygame.Color(0, 0, 0)


class Renderer:
    def __init__(self, game, title, sprites_path="../sprites", enable_render=True):
        self.game = game
        self.game_over = False
        self.enable_render = enable_render
        self.screen = None
        self.sprite_loader = None

        if self.enable_render:
            pygame.init()
            pygame.display.set_caption(title)
            self.screen_size = (
                game.GetMap().GetWidth() * BLOCK_SIZE,
                game.GetMap().GetHeight() * BLOCK_SIZE,
            )
            try:
                self.screen = pygame.display.set_mode(
                    (self.screen_size[0], self.screen_size[1]), pygame.DOUBLEBUF
                )
            except pygame.error as e:
                print(f"Error setting display mode: {e}")
                self.enable_render = False
                pygame.quit()
                return

            self.sprite_loader = sprites.SpriteLoader(sprites_path)

    def draw_obj(self, map, x_pos, y_pos):
        objects = map.At(x_pos, y_pos)

        for obj_type in objects.GetTypes():
            try:
                if pyBaba.IsTextType(obj_type):
                    if self.sprite_loader:
                        obj_image = self.sprite_loader.text_images[obj_type]
                    else:
                        continue
                else:
                    if obj_type == pyBaba.ObjectType.ICON_EMPTY:
                        continue

                    if self.sprite_loader:
                        obj_image = self.sprite_loader.icon_images[obj_type]
                    else:
                        continue

                obj_rect = obj_image.get_rect()
                obj_rect.topleft = (x_pos * BLOCK_SIZE, y_pos * BLOCK_SIZE)

                if self.screen:
                    self.screen.blit(obj_image, obj_rect)
            except KeyError:
                print(
                    f"Warning: Missing sprite for object type {obj_type} at ({x_pos}, {y_pos})"
                )
            except AttributeError as e:
                print(f"AttributeError during drawing object: {e}")

    def draw(self, map):
        if not self.enable_render or self.screen is None:
            return

        self.screen.fill(COLOR_BACKGROUND)

        for y_pos in range(map.GetHeight()):
            for x_pos in range(map.GetWidth()):
                self.draw_obj(map, x_pos, y_pos)

    def render(self, map, mode="human"):
        if not self.enable_render:
            self.process_event()
            return

        try:
            if not self.game_over:
                self.draw(map)

                if mode == "human":
                    pygame.display.flip()

            self.process_event()

        except Exception as e:
            print(f"Error during render/event processing: {e}")
            self.game_over = True
            self.quit_game()
            raise e

    def process_event(self):
        if not self.game_over:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                        self.quit_game()
            except pygame.error as e:
                if "pygame not initialized" in str(e):
                    print("Pygame not initialized, cannot process events.")
                    self.game_over = True
                else:
                    raise e

    def quit_game(self):
        self.game_over = True
        if self.enable_render:
            try:
                pygame.display.quit()
            except pygame.error as e:
                print(f"Error during pygame.display.quit(): {e}")
        try:
            pygame.quit()
        except pygame.error as e:
            print(f"Error during pygame.quit(): {e}")
