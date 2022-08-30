import pdb

import numpy as np
import gym
from src.model import Model_Actor_Critic

env_name = "HalfCheetah-v2"
env = gym.make(env_name)
seed = 0

np.random.seed(seed)
env.seed(seed)

actor_critic = Model_Actor_Critic(env.observation_space.shape[0],
                                       env.action_space.shape[0],
                                       [64, 64],
                                       [64, 64])

def collect_data(trajectories=100):  # evaluate
    data = np.array([np.array([0] * env.observation_space.shape[0]), np.array([0] * env.action_space.shape[0]), np.array([0] * env.observation_space.shape[0]), True, 0.0],
                    dtype=object)
    #actor_critic.load_weights(model_path)
    for e in range(trajectories):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        print("---Trajectory {}----",str(e))
        while True:
            #env.render()
            value, action, action_logp = actor_critic.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            exp = np.array([state, action, next_state, done, reward], dtype=object)
            data = np.vstack((data, exp))
            state = next_state
            if done:
                break
    np.save("datasets/rand_traj_Cheetah.npy", data)
    env.close()

collect_data(trajectories=200)