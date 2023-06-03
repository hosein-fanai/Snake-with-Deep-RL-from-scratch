from snake_environmnet import SnakeEnvironment

import gym


class SnakeGymEnvironment(gym.Env):

    def __init__(self, **kwargs):
        self.game = SnakeEnvironment(**kwargs)

        shape = (self.game.size, self.game.size)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=shape, dtype="uint8")
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        return self.game.step(action)

    def render(self, mode="rgb_array"):
        return self.game.render(mode)

    def reset(self):
        return self.game.reset()

    def close(self):
        self.game.close()