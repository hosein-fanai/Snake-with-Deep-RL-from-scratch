from utils import make_env

import sys

import time


if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
    except:
        # model_path = "models/Snake DQN.h5"
        model_path = "models/Snake A2C (stable-baselines3) rew_func1 (80mil iters).zip"

    if model_path[-3:] == ".h5":
        from dqn_agent import DQNAgent
        env = make_env(return_full_state=True)
        agent = DQNAgent(env, None, None, None, None, None)
        agent.load_model(model_path)
        agent.predict = lambda state: (agent.boltzman_sampling_policy(state), None)

    elif model_path[-4:] == ".zip":
        from stable_baselines3 import A2C
        env = make_env(num_stack=4)
        agent = A2C.load(model_path)

    state = env.reset()
    done = False
    rewards = 0

    frames = 0
    start = time.time()
    while not done:
        action, _ = agent.predict(state)
        state, reward, done, info = env.step(action)

        obs = env.render(mode="rgb_array")

        rewards += reward
        print(f"\r{rewards}, {reward}")

        frames += 1
        env.game.clock.tick(1/env.game.frame_time)
    env.close()

    fps = int(frames // (time.time() - start))
    print(f"Rewards: {rewards}, FPS:{fps}")