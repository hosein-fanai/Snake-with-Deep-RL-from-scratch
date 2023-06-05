import pygame

from snake import Snake

import numpy as np

import random

import time


class SnakeEnvironment(Snake):

    def __init__(self, size=15, return_full_state=False):
        super().__init__(size=size)
        self.return_full_state = return_full_state

    def _get_info(self):
        return {
            "score": self.score,
            "life": self.life,
            "head direction": self.direc
        }

    def _reward_func(self, info):
        reward = 0

        reward -= 0.1

        reward += 20 * (info["score"] - self.prev_info["score"])
        
        if info["life"] < self.prev_info["life"]:
            reward -= 50

        return reward

    def step(self, action):
        match action:
            case 0:
                self.key = None
            case 1:
                self.key = 'a'
            case 2:
                self.key = 'd'
            case 3:
                self.key = 'w'
            case 4:
                self.key = 's'
        self._move_to_key()
        
        info = self._get_info()
        reward = self._reward_func(info)
        done = False

        if self.life < 1:
            done = True

        self.prev_info = info

        state = self.arr.copy()

        if self.return_full_state:
            match info["head direction"]:
                case "right":
                    direc = 0
                case "left":
                    direc = 1
                case "up":
                    direc = 2
                case "down":
                    direc = 3
                    
            state = {
                "direc": direc,
                "board": state,
            }

        
        return state, reward, done, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            self._update_screen()
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
        elif mode == "print":
            self._print_display(0)
            return self.arr.copy()

    def reset(self):
        self.reset_game()

        self.prev_info = self._get_info()

        return self.arr.copy()

    def close(self):
        self.arr = None
        
        pygame.quit()


if __name__ == "__main__":
    env = SnakeEnvironment()
    state = env.reset()

    done = False
    frames = 0
    start = time.time()
    while not done:
        action = random.randint(0, 4)
        state, reward, done, info = env.step(action)

        env.render()

        frames += 1
        if frames > 1000:
            break

    fps = frames // (time.time() - start)
    print(f"FPS: {fps}")