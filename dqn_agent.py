import tensorflow as tf

import numpy as np

import time


class DQNAgent:

    def __init__(self, env, model, target, optimizer, loss_fn, replay_buffer):
        self.env = env
        self.model =  model
        self.target = target
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.replay_buffer = replay_buffer

        self.action_space_num = self.env.action_space.n

        if self.env.game.return_full_state:
            self._sample_experiences = self._sample_experiences2
        else:
            self._sample_experiences = self._sample_experiences1

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            state = {"direc": state["direc"][None], "board":state["board"][None]} if self.env.game.return_full_state else state[None]

            Q_values = self.model(state)
            return tf.argmax(Q_values[0])

    def sample_action(self, state):
        state = {"direc": state["direc"][None], "board":state["board"][None]} if self.env.game.return_full_state else state[None]

        Q_values = self.model(state)
        logits = tf.math.log(Q_values + tf.keras.backend.epsilon())
        action = tf.random.categorical(logits, num_samples=1)[0]

        return action

    def _play_one_step(self, state, epsilon=0.01):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = self.env.step(action)

        self.replay_buffer.append((state, action, reward, next_state, done))

        return next_state, reward, done, info

    def _play_multiple_steps(self, n_step, epsilon):
        obs = self.env.reset()
        total_reward = 0

        frames = 0
        start = time.time()
        for step in range(n_step):
            obs, reward, done, _ = self._play_one_step(obs, epsilon)
            total_reward += reward
            if done:
                break
            frames += 1
        fps = int(frames // (time.time() - start))
        
        return total_reward, step, fps

    def _func1(self, i):
        index = np.random.randint(len(self.replay_buffer), size=1)[0]
        exp = self.replay_buffer[index]

        return exp

    def _sample_experiences1(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array(sub_exp) for sub_exp in zip(*batch)
        ]

        return states, actions, rewards, next_states, dones

    def _sample_experiences2(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(range(batch_size))
        dataset = dataset.map(self._func1, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)

        return next(iter(dataset))

    @tf.function
    def _train_step(self, batch_size, gamma):
        exp_batch = self._sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = exp_batch

        next_Q_values = self.model(next_states)
        next_best_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(next_best_actions, depth=self.action_space_num)

        next_best_Q_values = tf.reduce_sum(self.target(next_states) * next_mask, axis=1)

        target_Q_values = rewards + (1.0 - tf.cast(dones, tf.float32)) * gamma * next_best_Q_values
        target_Q_values = tf.reshape(target_Q_values, (-1, 1))

        mask = tf.one_hot(actions, depth=self.action_space_num)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states, training=True)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def training_loop(self, iteration, n_step=10_000, batch_size=64, gamma=0.99, 
                    warmup=10, train_interval=1, target_update_interval=50, 
                    soft_update=False, epsilon_fn=None):
        func = lambda episode: max(1-episode/(iteration*0.9), 0.01)
        epsilon_fn = func if epsilon_fn is None else epsilon_fn

        best_score = float("-inf")
        rewards = []
        all_loss = []

        for episode in range(warmup):
            _, step, fps = self._play_multiple_steps(n_step, 1.0)

            print(f"\r---Warmup---Episode: {episode}, Steps: {step}, FPS: {fps}", end="")

        for episode in range(iteration):
            epsilon = epsilon_fn(episode)

            total_reward, step, fps = self._play_multiple_steps(n_step, epsilon)
            rewards.append(total_reward)

            if total_reward > best_score or total_reward > 5_000 or episode % 50_000 == 0:
                self.model.save_weights(f"models/DQN_ep#{episode}_eps#{epsilon:.4f}_rw#{total_reward:.1f}.h5")
                best_score = total_reward

            if episode % train_interval == 0:
                loss = self._train_step(batch_size, gamma)
            all_loss.append(loss)

            if episode % target_update_interval == 0:
                if not soft_update:
                    self.target.set_weights(self.model.get_weights())
                else:
                    target_weights = self.target.get_weights()
                    online_weights = self.model.get_weights()
                    for index in range(len(target_weights)):
                        target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                    self.target.set_weights(target_weights)

            print(f"\rEpisode: {episode}, Steps: {step}, FPS: {fps}, Reward:{total_reward:.1f}, Epsilon: {epsilon:.4f}, Loss: {loss}", end="")

        return rewards, all_loss