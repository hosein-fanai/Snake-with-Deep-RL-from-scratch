from snake_environmnet import SnakeEnvironment

import gym
from gym.wrappers import FrameStack


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
        

class MaxStepsWrapper(gym.Wrapper):
        
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps

        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        
        truncate = False
        if self.current_step > self.max_steps:
            truncate = True

        state, reward, done, info = self.env.step(action)
        done = done or truncate
        
        return state, reward, done, info
    
    
def make_env(size=10, return_full_state=False, max_step=0, num_stack=0):
    env = SnakeGymEnvironment(size=size, return_full_state=return_full_state)
    
    if max_step:
        env = MaxStepsWrapper(env, max_step)
    
    if num_stack:
        env = FrameStack(env, num_stack=4)

    return env