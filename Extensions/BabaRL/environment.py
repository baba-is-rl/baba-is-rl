import gym
import numpy as np
import pyBaba
from gym.envs.registration import register
from gym.utils import seeding

import rendering


def register_env(env_id: str, path: str, display_title: str):
    register(
        id=env_id,
        entry_point="environment:BabaEnv",
        max_episode_steps=200,
        nondeterministic=True,
        kwargs={"path": path, "display_title": display_title},
    )


class BabaEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, path, display_title, enable_render=True):
        super(BabaEnv, self).__init__()

        self.path = path
        self.game = pyBaba.Game(self.path)
        self.renderer = rendering.Renderer(self.game, display_title)

        self.action_space = [
            pyBaba.Direction.UP,
            pyBaba.Direction.DOWN,
            pyBaba.Direction.LEFT,
            pyBaba.Direction.RIGHT,
        ]

        self.action_size = len(self.action_space)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        self.game.Reset()
        self.done = False

        return self.get_obs()

    def step(self, action):
        self.game.MovePlayer(action)

        result = self.game.GetPlayState()

        if result == pyBaba.PlayState.LOST:
            self.done = True
            reward = -100
        elif result == pyBaba.PlayState.WON:
            self.done = True
            reward = 200
        else:
            reward = -0.5

        return self.get_obs(), reward, self.done, {}

    def render(self, mode="human", close=False):
        if close:
            self.renderer.quit_game()

        return self.renderer.render(self.game.GetMap(), mode)

    def get_obs(self):
        return np.array(
            pyBaba.Preprocess.StateToTensor(self.game), dtype=np.float32
        ).reshape(-1, self.game.GetMap().GetHeight(), self.game.GetMap().GetWidth())
