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
        head_pos = np.where(self.arr == 1)
        food_pos = np.where(self.arr == 4)
        head_food_dist = np.abs(head_pos[0] - food_pos[0]) + np.abs(head_pos[1] - food_pos[1])

        return {
            "score": self.score,
            "life": self.life,
            "head direction": self.direc,
            "head food dist": head_food_dist,
            "head pos": head_pos,
            "size": self.size,
        }

    def _reward_func(self, info, prev_info):
        reward = 0

        # if info["head food dist"] < prev_info["head food dist"]:
        #     reward += 0.2
        # else:
        #     reward -= 0.1

        # if (prev_info["head pos"][0] == info["size"]-1 and info["head pos"][0] == 0) or (
        #     prev_info["head pos"][1] == info["size"]-1 and info["head pos"][1] == 0) or (
        #     prev_info["head pos"][0] == 0 and info["head pos"][0] == info["size"]-1) or (
        #     prev_info["head pos"][1] == 0 and info["head pos"][1] == info["size"]-1):
        #     reward -= 1

        # reward -= 0.1

        reward += 1 * (info["score"] - prev_info["score"])
        
        if info["life"] < prev_info["life"]:
            reward -= 1

        return reward

    def compute_reward(self, info):
        return self._reward_func(info, self.prev_info)

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
        reward = self.compute_reward(info)
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
                "direc": np.array(direc),
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

        state = self.arr.copy()

        if self.return_full_state:
            match self.prev_info["head direction"]:
                case "right":
                    direc = 0
                case "left":
                    direc = 1
                case "up":
                    direc = 2
                case "down":
                    direc = 3
                    
            state = {
                "direc": np.array(direc),
                "board": state,
            }

        return state

    def close(self):
        self.arr = None
        
        pygame.quit()


if __name__ == "__main__":
    env = SnakeEnvironment(size=10)

    state = env.reset()

    done = False
    rewards = 0

    frames = 0
    start = time.time()
    while not done:
        action = random.randint(0, 4)
        # action = int(input())
        state, reward, done, info = env.step(action)

        rewards += reward
        print(rewards, reward)

        env.render("rgb_array")

        frames += 1
        if frames > 1000:
            break

        # env.clock.tick(10)

    fps = frames // (time.time() - start)
    print(f"FPS: {fps}")