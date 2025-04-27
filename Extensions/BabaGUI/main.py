import os
import sys

import pyBaba
import pygame

import config
import sprites

MAP_PATH = "../../Resources/Maps/priming/lvl6.txt"
SPRITES_PATH = "./sprites"

icon_images_map = {
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
    pyBaba.ObjectType.ICON_EMPTY: None,
}

text_images_map = {
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

pygame.init()
pygame.font.init()

try:
    game = pyBaba.Game(MAP_PATH)
except Exception as e:
    print(f"Error loading map '{MAP_PATH}': {e}")
    sys.exit(1)

screen_size = (
    game.GetMap().GetWidth() * config.BLOCK_SIZE,
    game.GetMap().GetHeight() * config.BLOCK_SIZE,
)

try:
    screen = pygame.display.set_mode((screen_size[0], screen_size[1]), pygame.DOUBLEBUF)
    pygame.display.set_caption("Baba Is GUI")
except pygame.error as e:
    print(f"Error setting display mode ({screen_size[0]}x{screen_size[1]}): {e}")
    print("Running in headless mode. Rendering will be skipped.")
    screen = None


sprite_loader = None
if screen:
    try:
        if not os.path.exists(SPRITES_PATH):
            raise FileNotFoundError(f"Sprites path not found: {SPRITES_PATH}")
        sprite_loader = sprites.SpriteLoader(SPRITES_PATH)
        print(f"Loaded sprites from: {SPRITES_PATH}")
    except Exception as e:
        print(f"Error loading sprites: {e}")
        sprite_loader = None

map_sprite_group = pygame.sprite.Group()

result_image = None
result_image_group = None
if screen and sprite_loader and sprite_loader.result_images_loaded:
    result_image = sprites.ResultImage(sprite_loader)
    result_image_group = pygame.sprite.GroupSingle(result_image)


def draw_obj(x_pos, y_pos):
    if not screen or not sprite_loader:
        return

    map_object = game.GetMap().At(x_pos, y_pos)
    objects_to_draw = []

    for obj_type in map_object.GetTypes():
        img_surface = None
        priority = sprites.DEFAULT_PRIORITY

        if pyBaba.IsTextType(obj_type):
            if obj_type in sprite_loader.text_images:
                img_surface = sprite_loader.text_images[obj_type]
                priority = sprites.get_layer_priority(obj_type)
        elif obj_type != pyBaba.ObjectType.ICON_EMPTY:
            if obj_type in sprite_loader.icon_images:
                img_surface = sprite_loader.icon_images[obj_type]
                priority = sprites.get_layer_priority(obj_type)

        if img_surface and priority >= 0:
            img_rect = img_surface.get_rect(
                topleft=(x_pos * config.BLOCK_SIZE, y_pos * config.BLOCK_SIZE)
            )
            objects_to_draw.append((priority, img_surface, img_rect))

    objects_to_draw.sort(key=lambda item: item[0])

    for _, surface, rect in objects_to_draw:
        try:
            screen.blit(surface, rect)
        except pygame.error as e:
            print(f"Error blitting sprite at {rect.topleft}: {e}")


def draw():
    if not screen or not sprite_loader:
        return

    screen.fill(config.COLOR_BACKGROUND)
    map_height = game.GetMap().GetHeight()
    map_width = game.GetMap().GetWidth()

    for y_pos in range(map_height):
        for x_pos in range(map_width):
            draw_obj(x_pos, y_pos)


if __name__ == "__main__":
    clock = pygame.time.Clock()
    game_over = False
    running = True

    while running:
        player_action = pyBaba.Direction.NONE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    print("Resetting level...")
                    try:
                        game.Reset()
                        game_over = False
                    except Exception as e:
                        print(f"Error during game reset: {e}")

                if not game_over:
                    if event.key == pygame.K_UP:
                        player_action = pyBaba.Direction.UP
                    elif event.key == pygame.K_DOWN:
                        player_action = pyBaba.Direction.DOWN
                    elif event.key == pygame.K_LEFT:
                        player_action = pyBaba.Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        player_action = pyBaba.Direction.RIGHT
                    elif event.key == pygame.K_SPACE:
                        player_action = pyBaba.Direction.NONE

        if not game_over and player_action != pyBaba.Direction.NONE:
            try:
                game.MovePlayer(player_action)
                current_state = game.GetPlayState()
                if (
                    current_state == pyBaba.PlayState.WON
                    or current_state == pyBaba.PlayState.LOST
                ):
                    game_over = True
                    print(f"Game Over! State: {current_state.name}")
                    if result_image_group and result_image:
                        result_image.update(current_state, screen_size)

            except Exception as e:
                print(f"Error during game.MovePlayer({player_action.name}): {e}")
                game_over = True
                running = False

        if screen:
            draw()

            if game_over and result_image_group:
                result_image_group.draw(screen)

            pygame.display.flip()

        clock.tick(config.FPS)

    pygame.quit()
    print("Exiting BabaGUI.")
    sys.exit()
