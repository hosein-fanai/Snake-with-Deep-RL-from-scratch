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
    
    
def rwd_func_1(info, prev_info):
    reward = 0

    reward += 1 * (info["score"] - prev_info["score"])

    return reward

def rwd_func_2(info, prev_info):
    reward = 0

    reward -= 0.1

    reward += 1.1 * (info["score"] - prev_info["score"])
    
    if info["life"] < prev_info["life"]:
        reward -= 0.9

    return reward

def rwd_func_3(info, prev_info):
    reward = 0

    if (prev_info["head pos"][0] == info["size"]-1 and info["head pos"][0] == 0) or (
        prev_info["head pos"][1] == info["size"]-1 and info["head pos"][1] == 0) or (
        prev_info["head pos"][0] == 0 and info["head pos"][0] == info["size"]-1) or (
        prev_info["head pos"][1] == 0 and info["head pos"][1] == info["size"]-1):
        reward -= 0.5

    reward -= 0.1

    reward += 1.1 * (info["score"] - prev_info["score"])
    
    if info["life"] < prev_info["life"]:
        reward -= 0.9

    return reward

def rwd_func_4(info, prev_info):
    reward = 0

    if info["head food dist"] < prev_info["head food dist"]:
        reward += 0.2
    else:
        reward -= 0.1

    if (prev_info["head pos"][0] == info["size"]-1 and info["head pos"][0] == 0) or (
        prev_info["head pos"][1] == info["size"]-1 and info["head pos"][1] == 0) or (
        prev_info["head pos"][0] == 0 and info["head pos"][0] == info["size"]-1) or (
        prev_info["head pos"][1] == 0 and info["head pos"][1] == info["size"]-1):
        reward -= 0.5

    reward -= 0.1

    reward += 1.1 * (info["score"] - prev_info["score"])
    
    if info["life"] < prev_info["life"]:
        reward -= 0.9

    return reward


def make_env(size=10, env_rwd_func=None, return_full_state=False, max_step=0, num_stack=0):
    env = SnakeGymEnvironment(size=size, return_full_state=return_full_state)

    if env_rwd_func:
        env.game._reward_func = env_rwd_func
    
    if max_step:
        env = MaxStepsWrapper(env, max_step)
    
    if num_stack:
        env = FrameStack(env, num_stack=4)

    return env