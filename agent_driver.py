from tensorflow.keras import models

from utils import SnakeGymEnvironment, DQNAgent

import sys

import time


if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
    except:
        model_path = "models/Snake DQN.h5"

    env = SnakeGymEnvironment()

    model = models.load_model(model_path)
    agent = DQNAgent(env, model, None, None, None, None)
    
    state = env.reset()
    done = False
    rewards = 0

    frames = 0
    start = time.time()
    while not done:
        action = agent.epsilon_greedy_policy(obs)
        obs, reward, done, info = env.step(action)
        rewards += reward

        env.game.clock.tick(1/env.game.frame_time)
        frames += 1
    env.close()

    fps = int(frames // (time.time() - start))
    print(f"Rewards: {rewards}, FPS:{fps}")