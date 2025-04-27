import os

import gym
import gym.spaces as spaces
import numpy as np
import pyBaba
from gym.envs.registration import register
from gym.utils import seeding

import rendering
from constants import (
    MAX_MAP_HEIGHT,
    MAX_MAP_WIDTH,
    MAX_RULES,
    NUM_OBJECT_TYPES,
    PAD_INDEX,
)


def register_env(
    env_id: str,
    map_name: str,
    display_title: str,
    maps_dir: str = "../../../Resources/Maps",
    sprites_path: str = "../sprites",
    enable_render: bool = True,
):
    map_path = os.path.join(maps_dir, map_name)
    register(
        id=env_id,
        entry_point="environment:BabaEnv",
        max_episode_steps=200,
        nondeterministic=True,
        kwargs={
            "map_path": map_path,
            "display_title": display_title,
            "sprites_path": sprites_path,
            "enable_render": enable_render,
        },
    )


class BabaEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, map_path, display_title, sprites_path, enable_render=True):
        super(BabaEnv, self).__init__()

        self.map_path = map_path
        self.game = pyBaba.Game(self.map_path)
        self.renderer = rendering.Renderer(
            self.game, display_title, sprites_path, enable_render
        )
        self.enable_render = enable_render

        self.action_space_list = [
            pyBaba.Direction.UP,
            pyBaba.Direction.DOWN,
            pyBaba.Direction.LEFT,
            pyBaba.Direction.RIGHT,
            # pyBaba.Direction.NONE,
        ]
        self.action_space = spaces.Discrete(len(self.action_space_list))

        self.grid_shape = (pyBaba.Preprocess.TENSOR_DIM, MAX_MAP_HEIGHT, MAX_MAP_WIDTH)
        self.rules_shape = (MAX_RULES, 3)  # Noun, Operator, Target indices
        self.rule_mask_shape = (MAX_RULES,)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0, high=1.0, shape=self.grid_shape, dtype=np.float32
                ),
                "rules": spaces.Box(
                    low=0,
                    high=NUM_OBJECT_TYPES - 1,
                    shape=self.rules_shape,
                    dtype=np.int64,
                ),
                "rule_mask": spaces.Box(
                    low=0, high=1, shape=self.rule_mask_shape, dtype=np.int64
                ),  # 0=pad, 1=valid
            }
        )
        # -----------------------------

        self.current_map_width = self.game.GetMap().GetWidth()
        self.current_map_height = self.game.GetMap().GetHeight()

        self.seed()
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        raw_grid_tensor_list = pyBaba.Preprocess.StateToTensor(self.game)
        raw_grid_np = np.array(raw_grid_tensor_list, dtype=np.float32).reshape(
            pyBaba.Preprocess.TENSOR_DIM,
            self.current_map_height,
            self.current_map_width,
        )

        padded_grid = np.zeros(self.grid_shape, dtype=np.float32)
        h = min(self.current_map_height, MAX_MAP_HEIGHT)
        w = min(self.current_map_width, MAX_MAP_WIDTH)
        padded_grid[:, :h, :w] = raw_grid_np[:, :h, :w]

        rules_list = pyBaba.Preprocess.GetAllRules(self.game)
        rules_tensor = np.full(self.rules_shape, PAD_INDEX, dtype=np.int64)
        rule_mask = np.zeros(self.rule_mask_shape, dtype=np.int64)

        for i, rule in enumerate(rules_list):
            if i >= MAX_RULES:
                # print(f"Warning: Exceeded MAX_RULES ({MAX_RULES}). Truncating rule list.") # Optional warning
                break

            obj1, obj2, obj3 = rule.GetObjects()

            type1 = (
                obj1.GetTypes()[0] if obj1.GetTypes() else pyBaba.ObjectType.ICON_EMPTY
            )
            type2 = (
                obj2.GetTypes()[0] if obj2.GetTypes() else pyBaba.ObjectType.ICON_EMPTY
            )  # Should be IS/HAS etc.
            type3 = (
                obj3.GetTypes()[0] if obj3.GetTypes() else pyBaba.ObjectType.ICON_EMPTY
            )

            rules_tensor[i, 0] = int(type1.value)
            rules_tensor[i, 1] = int(type2.value)
            rules_tensor[i, 2] = int(type3.value)
            rule_mask[i] = 1

        return {"grid": padded_grid, "rules": rules_tensor, "rule_mask": rule_mask}

    def reset(self, new_map_path=None):
        if new_map_path and new_map_path != self.map_path:
            print(f"Loading new map: {new_map_path}")
            self.map_path = new_map_path
            try:
                self.game = pyBaba.Game(self.map_path)
                self.current_map_width = self.game.GetMap().GetWidth()
                self.current_map_height = self.game.GetMap().GetHeight()

                if self.enable_render and self.renderer:
                    self.renderer.update_game_reference(self.game)
                    self.renderer.update_display_size(
                        self.current_map_width, self.current_map_height
                    )

            except Exception as e:
                print(f"Error loading map {new_map_path}: {e}")
                raise
        else:
            self.game.Reset()

        self.done = False
        return self._get_obs()

    def step(self, action_index):
        if self.done:
            # print("Warning: step() called after environment is done.")
            return self._get_obs(), 0.0, True, {}

        action = self.action_space_list[action_index]

        self.game.MovePlayer(action)
        result = self.game.GetPlayState()

        # Reward Structure
        reward = 0.0
        if result == pyBaba.PlayState.LOST:
            self.done = True
            reward = -1.0
        elif result == pyBaba.PlayState.WON:
            self.done = True
            reward = 1.0
        else:
            self.done = False
            reward = -0.01

        return self._get_obs(), reward, self.done, {}

    def render(self, mode="human", close=False):
        if not self.enable_render:
            return None

        if close:
            if self.renderer:
                self.renderer.quit_game()
            return None

        if self.renderer:
            return self.renderer.render(self.game.GetMap(), mode)
        else:
            # print("Warning: Renderer not available for rendering.")
            return None

    def close(self):
        """Clean up resources."""
        if self.enable_render and self.renderer:
            self.renderer.quit_game()

