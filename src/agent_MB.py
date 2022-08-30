import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore general tf warnings
from collections import deque
import pickle
import matplotlib.pyplot as plt
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import plot_util as pu

from src.env_wrappers import VecNormalize, TimeLimitMask
from src.model import Model_Actor_Critic

global NUM_TRAIN_UPDATES
global MINIBATCH_SIZE


class PPO_Agent():
    def __init__(self, params, env_name, seed=0,
        max_trajectories = 10, short_horizon = 10, title = "Default", save_prefix = "",
        logger_neptune = None, last_save = 0, MPC_only = False, reduced_obs = False, mpc_rand = False):
        self.params = params
        self.save_prefix = save_prefix
        self.env_name = env_name
        self.seed = seed

        # Construct agent
        self.construct_agent()
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.shape[0]

        # Create log directory
        os.makedirs(os.path.join(self.save_prefix, "Logging", self.env_name.split('-')[0]), exist_ok=True)

        # Construct testing environment - bench.Monitor used for logging episodes, VecNormalize for normalizing states & rewards
        self.env = bench.Monitor(self.env, os.path.join(self.save_prefix, "Logging", self.env_name.split('-')[0],
                                                        '0'),  allow_early_resets=True)  # FILE PATH MUST HAVE A DIGIT AT END  ,
        self.env = DummyVecEnv([lambda: self.env])  # Needed to use the VecNormalize wrapper
        self.env = VecNormalize(self.env,
                                ob=True, ret=True,
                                gamma=self.params['GAMMA'])  # Normalize states AND rewards for training

        self.update = 1
        self.episode_reward_summary = deque(maxlen=10)
        '''MPC config'''
        self.logger = logger_neptune
        self.max_trajectories = max_trajectories
        self.short_horizon = short_horizon
        self.title = title

        self.last_save = last_save
        self.MPC_only = MPC_only
        self.reduced_obs = reduced_obs
        self.start_ac = np.zeros((self.max_trajectories, self.action_size))
        self.start_val = np.zeros(self.max_trajectories)
        self.start_logp = np.zeros(self.max_trajectories)
        self.horizon_rew = np.zeros(self.max_trajectories)
        self.PPO_tuning = self.params["PPO_Tuning"]
        global NUM_TRAIN_UPDATES, MINIBATCH_SIZE
        NUM_TRAIN_UPDATES = int(self.params['NUM_ENV_TIMESTEPS']) // self.params['NUM_TIMESTEPS_PER_UPDATE']
        MINIBATCH_SIZE = int(self.params['NUM_TIMESTEPS_PER_UPDATE']) // self.params['NUM_MINIBATCHES']

        if mpc_rand:
            self.MPC_call = self.MPC_multistep
        else:
            self.MPC_call = self.MPC_randActions

    def construct_agent(self):
        self.env = gym.make(self.env_name,  exclude_current_positions_from_observation = False)  # Note: Do not unwrap, only need general attributes of env

        # Set seed for deterministic results
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.env.seed(self.seed)

        # TimeLimit wrapper useful for Mujoco environments
        self.env = TimeLimitMask(self.env)

        # Models - training step function, networks, and optimizer must be redefined/reconstructed after switching environments
        self.actor_critic = Model_Actor_Critic(self.env.observation_space.shape[0]-1,
                                               self.env.action_space.shape[0],
                                               self.params['ACTOR_HIDDEN_UNITS'],
                                               self.params['CRITIC_HIDDEN_UNITS'])
        self.train_model = self.get_train_model_function()
        self.optimizer = Adam(learning_rate=self.params['LEARNING_RATE'], epsilon=self.params['OPTIMIZER_EPSILON'])
        self.get_action = self.actor_critic.act


    def set_env(self, env_name, **kwargs):  # 'seed' and 'params' are optional arguments
        self.env_name = env_name
        self.params = kwargs.get('params', self.params)
        self.seed = kwargs.get('seed', self.seed)

    def calculate_returns(self, rewards, masks, bad_masks, values):
        # Use generalized advantage estimator (balance of bias & variance with lambda-return compared to TD and MC)
        # If lambda = 1, then becomes MC. If lambda is 0, then becomes TD
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.params['GAMMA'] * values[i + 1] * masks[i] - values[i]
            gae = delta + self.params['GAMMA'] * self.params['LAMBDA'] * masks[i] * gae
            gae = gae * bad_masks[i]  # If not a true transition due to environment reset, set gae to 0
            returns.insert(0, gae)  # Keep inserting at beginning because we're traversing in reverse order
        returns += values[:-1]
        return returns

    # Returns a new function whenever environments are switched because tf.function creates a graph specific to the function
    def get_train_model_function(self):
        # @tf.function decoration runs function in graph mode - signficantly faster than eager execution in tf 2.0
        @tf.function
        def train_model(states, actions, returns, old_values, old_logps):
            advantages = returns - old_values
            mean, var = tf.nn.moments(advantages, [0], keepdims=True)  # Standardize advantages in each MINIBATCH
            advantages = (advantages - mean) / (tf.sqrt(var) + 1e-8)

            with tf.GradientTape() as tape:
                values, logps, dist_entropy = self.actor_critic.evaluate_actions(states, actions)

                ratio = tf.exp(logps - old_logps)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.params['CLIP_PARAM'], 1 + self.params['CLIP_PARAM'])

                actor_loss = tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                critic_loss = tf.reduce_mean(0.5 * tf.square(returns - values))

                loss = -actor_loss + (self.params['VALUE_FUNCTION_COEF'] * critic_loss) - (
                            self.params['ENTROPY_COEF'] * dist_entropy)

            gradients = tape.gradient(loss, self.actor_critic.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_weights))

        return train_model

    def train(self, PPO_updates = 10, update_counter = 0, model = None):

        #data = np.array([np.array([0] * self.state_size[0]), np.array([0] * self.env.action_space.shape[0]), np.array([0] * self.state_size[0]), True, 0.0], dtype=object)

        state = self.env.reset()  # Reset once at beginning, all subsequent resets handled by monitor
        while self.update < NUM_TRAIN_UPDATES + 1:
            start_time = time.time()
            if self.PPO_tuning:
                train_low = -1
                train_high = -1
            else:
                if self.update % 2 == 0 and self.update > 1:
                    train_low = 500
                    train_high = 1000
                else:
                    train_low = 0
                    train_high = 500

            states = [state]* self.params['NUM_TIMESTEPS_PER_UPDATE']
            actions = [np.zeros(self.env.action_space.shape[0])]*self.params['NUM_TIMESTEPS_PER_UPDATE']
            rewards = [0.0]*self.params['NUM_TIMESTEPS_PER_UPDATE']
            masks = [0.0]*self.params['NUM_TIMESTEPS_PER_UPDATE']
            bad_masks = [0.0]*self.params['NUM_TIMESTEPS_PER_UPDATE']
            values = [[0.0]]*(self.params['NUM_TIMESTEPS_PER_UPDATE']+1)
            old_logps = [[0.0]]*self.params['NUM_TIMESTEPS_PER_UPDATE']

            """Perform Rollout"""
            for update_step in range(1, self.params['NUM_TIMESTEPS_PER_UPDATE'] + 1):
                if train_low < update_step <= train_high:
                    value, action, logp_t = self.MPC_call(state,model)
                else:
                    value, action, logp_t = self.get_action(state[1:])
                next_state, reward, done, info = self.env.step(action)

                if 'episode' in info.keys():
                    self.episode_reward_summary.append(info['episode']['r'])  # Logged by bench.Monitor

                states[update_step-1] = state[1:]
                actions[update_step-1] = action
                rewards[update_step-1] = reward
                old_logps[update_step-1] = logp_t
                values[update_step-1] = value
                masks[update_step-1] = (0.0 if done else 1.0)
                bad_masks[update_step-1] = (0.0 if 'bad_transition' in info.keys() else 1.0) # Occurs when a done is the result
                # of exceeding the environment step limit
                #exp = np.array([np.reshape(state, [self.state_size[0]]), action, next_state, done, reward],
                               #dtype=object)
                #data = np.vstack((data, exp))
                state = next_state

            values[-1] = self.actor_critic.get_value(next_state[1:])

            """Update Networks"""
            states = np.asarray(states, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)
            bad_masks = np.asarray(bad_masks, dtype=np.float32)
            values = np.asarray(values, dtype=np.float32)
            old_logps = np.asarray(old_logps, dtype=np.float32)

            # Caclulate returns
            returns = self.calculate_returns(rewards, masks, bad_masks, values)

            # Create minibatches and train model
            inds = np.arange(self.params['NUM_TIMESTEPS_PER_UPDATE'])
            for _ in range(self.params['NUM_EPOCHS']):
                np.random.shuffle(
                    inds)  # IMPORTANT: shuffling data indices for training stability. Since this problem is a sequential task,
                # consecutive samples are very similar to each other in unshuffled data. Our minibatch sizes are already
                # fairly small, so not shuffling could lead us to have conflicting gradients for each minibatch.
                for start in range(0, self.params['NUM_TIMESTEPS_PER_UPDATE'], MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    mb = (arr[mb_inds] for arr in (states, actions, returns, values, old_logps))

                    self.train_model(*mb)

            end_time = time.time()
            elapsed_time = end_time - start_time

            """ Print Summary """
            print(
                "Update {}/{}, Timesteps completed {}\nLast {} episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, elapsed update time {:.3f} seconds\n"
                .format(self.update, NUM_TRAIN_UPDATES, int(self.params['NUM_TIMESTEPS_PER_UPDATE'] * self.update),
                        len(self.episode_reward_summary),
                        np.mean(self.episode_reward_summary), np.median(self.episode_reward_summary),
                        np.min(self.episode_reward_summary),
                        np.max(self.episode_reward_summary), elapsed_time))
            if self.logger:
                self.logger["Mean reward"].log(np.mean(self.episode_reward_summary), self.params['NUM_TIMESTEPS_PER_UPDATE'] * self.update)
            l = open(self.save_prefix + "/Mean_rewards.txt", "a+")
            l.write("Timestep %d    " % int(self.params['NUM_TIMESTEPS_PER_UPDATE'] * self.update))
            l.write("Minimum %d     " % np.min(self.episode_reward_summary))
            l.write("Maximum %d     " % np.max(self.episode_reward_summary))
            l.write("Average %d\r\n" % np.mean(self.episode_reward_summary))
            l.close()
            k =  open(self.save_prefix + "/Episode_reward.txt", "a+")
            k.write("Episode %d    " % len(self.episode_reward_summary))
            k.write("Score %d\r\n" % self.episode_reward_summary[-1])
            k.close()
            #np.save(os.path.join(self.save_prefix, "PPO_data{}.npy".format(self.update)),data, allow_pickle=True)
            #data = np.array([np.array([0] * self.state_size[0]), np.array([0] * self.env.action_space.shape[0]),
                             #np.array([0] * self.state_size[0]), True, 0.0], dtype=object)

            if (self.update % self.params['SAVE_INTERVAL'] == 0 or self.update == NUM_TRAIN_UPDATES):
                # Save model weights
                model_path = os.path.join(self.save_prefix, "Models", 'model_weights{}'.format(self.update))
                self.actor_critic.save_weights(model_path)

                # Save ob_rms from enviornment via pickle so that it can be restored when testing
                vecnorm_path = os.path.join(self.save_prefix, "Scalars", 'vecnorm_stats{}.pickle'.format(self.update))
                with open(vecnorm_path, 'wb') as handle:
                    pickle.dump(getattr(self.env, 'ob_rms', None), handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.update = self.update + 1
            if self.update-1 % PPO_updates == 0:
                return []
        # Done with environment, can close
        self.env.close()

    def load(self, load_update):
        # Reconstruct environment, with NON-NORMALIZED REWARDS
        self.env = DummyVecEnv([lambda: self.env])  # Needed to use the VecNormalize wrapper
        self.env = VecNormalize(self.env, ob=True, ret=False)

        # Restore model weights
        model_path = os.path.join(self.save_prefix, "Models", 'model_weights{}'.format(load_update))
        self.actor_critic.load_weights(model_path)

        # Restored saved ob_rms to environment, disable updates to ob_rms
        saved_ob_rms = None
        vecnorm_path = os.path.join(self.save_prefix, "Scalars", 'vecnorm_stats{}.pickle'.format(load_update))
        with open(vecnorm_path, 'rb') as handle:
            saved_ob_rms = pickle.load(handle)
        self.update = load_update+1
        if saved_ob_rms is not None:
            self.env.ob_rms = saved_ob_rms




    def test(self, load_update):
        # Construct agent
        self.construct_agent()

        # Reconstruct environment, with NON-NORMALIZED REWARDS
        self.env = DummyVecEnv([lambda: self.env])  # Needed to use the VecNormalize wrapper
        self.env = VecNormalize(self.env, ob=True, ret=False)

        # Restore model weights
        model_path = os.path.join(self.save_prefix, "Models", 'model_weights{}'.format(load_update))
        self.actor_critic.load_weights(model_path)

        # Restored saved ob_rms to environment, disable updates to ob_rms
        saved_ob_rms = None
        vecnorm_path = os.path.join(self.save_prefix, "Scalars", 'vecnorm_stats{}.pickle'.format(load_update))
        with open(vecnorm_path, 'rb') as handle:
            saved_ob_rms = pickle.load(handle)

        if saved_ob_rms is not None:
            self.env.ob_rms = saved_ob_rms
        self.env.eval()

        state = self.env.reset()  # Reset once at beginning, all subsequent resets handled by monitor

        """ Only performing rollout for testing """
        score = 0
        ep_len = 0
        while True:  # Indefinitely performs rollout in simulator until closed
            self.env.render()

            _, action, _ = self.actor_critic.act(state[1:])
            next_state, rew, done, info = self.env.step(action)
            score += rew
            ep_len = 0
            if done:
                l = open(self.save_prefix + "/Test_results.txt", "a+")
                l.write("Score %d\r\n    " % score)
                l.close()
                score = 0
                ep_len = 0

            state = next_state

        # Done with environment, can close
        self.env.close()

    def plot_results(self):
        # Create plot directory
        os.makedirs(self.save_prefix, "Plots", exist_ok=True)

        results = pu.load_results(os.path.join( self.save_prefix, "Logging", self.env_name.split('-')[0], ''))
        pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')

        fig = plt.gcf()
        plot_path = os.path.join(self.save_prefix, "Plots", 'plot_' + self.env_name)
        fig.savefig(plot_path, bbox_inches='tight')

        plt.show()

    def MPC_action_selector(self, state, x_pos, model):
        start_state = np.append(state, x_pos)
        for traj_num in range(self.max_trajectories):
            _ = model.reset(start_state)
            # env_test = copy.deepcopy(self.env)
            value, action, logp_t  = self.actor_critic.act(state)
            # ac = np.random.choice(self.action_range, self.start_ac.shape[1])
            self.start_val[traj_num] = value
            self.start_ac[traj_num] = action
            self.start_logp[traj_num] = logp_t
            cum_rew = 0
            for _ in range(self.short_horizon):
                model_state, rew, done, _ = model.step(action=action)
                # state, rew, done, _ = env_test.step(action)
                if done:
                    break
                cum_rew += rew
                _, action, _ = self.actor_critic.act(model_state[1:])
            self.horizon_rew[traj_num] = cum_rew  # rewrites old values, no need to clear
        '''return best action only'''
        elite_idx = self.horizon_rew.argsort()[0]
        return self.start_val[elite_idx], self.start_ac[elite_idx], self.start_logp[elite_idx]


    def MPC_multistep(self, state, model):
        States = np.array([state] * self.max_trajectories)
        StatesAcc = np.array([States]*self.short_horizon)
        Next_StatesAcc = np.array([[state] * self.max_trajectories] * self.short_horizon)
        Values, Actions, Logp_ts = map(np.array, zip(*[self.get_action(state[1:]) for _ in range(self.max_trajectories)]))
        Start_Actions = Actions.copy()
        ActionAcc = np.array([Actions]* self.short_horizon)
        for i in range(self.short_horizon):
            if i>0:
                StatesAcc[i] = States
                ActionAcc[i] = Actions
            Next_StatesAcc[i] = Next_States = model.multi_predict(States, Actions)
            States = Next_States
            Actions = np.array([self.get_action(States[i,1:])[1] for i in range(self.max_trajectories)])
        scores = model.Scores(StatesAcc, Next_StatesAcc, ActionAcc)
        elite_idx = scores.argsort()[-1]
        return Values[elite_idx], Start_Actions[elite_idx], Logp_ts[elite_idx]

    def MPC_randActions(self, state, model):
        States = np.array([state] * self.max_trajectories)
        StatesAcc = np.array([States]*self.short_horizon)
        Next_StatesAcc = np.array([[state] * self.max_trajectories] * self.short_horizon)
        Actions = np.array(
            [self.env.action_space.sample() for p in range(self.short_horizon * self.max_trajectories)]).reshape(
            (self.short_horizon, self.max_trajectories, -1))

        for i in range(self.short_horizon):
            if i>0:
                StatesAcc[i] = States
            Next_StatesAcc[i] = Next_States = model.multi_predict(States, Actions[i,:])
            States = Next_States
        scores = model.Scores(StatesAcc, Next_StatesAcc, Actions)
        elite_idx = scores.argsort()[-1]
        action = Actions[0, elite_idx]
        value, logp_t, _ = self.actor_critic.evaluate_actions(StatesAcc[0,elite_idx].reshape([1,-1]), action)
        return value.numpy()[0], Actions[0,elite_idx], logp_t.numpy()[0]