import os
import sys

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from ModelBase import ModelBase
from ReplayBuffer import ReplayMemoryFast
from src.agent_MB import PPO_Agent
import copy
from Result_Plotter import Plotter
from variables import Variables
import neptune.new as neptune

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def reduce_state(data):
    new_data = data
    for i in range(0,data.shape[0]):
        exp = np.array([data[i][0][:-10], data[i][1], data[i][2][:-10], data[i][3]], dtype = object)
        new_data[i] = exp
    return new_data

def train(configuration, logger_neptune, seed = 0):
    MPC_collect = 0
    np.random.seed(seed)

    Drand = ReplayMemoryFast(memory_size=60000,
                             state_size=configuration["ModelBase"]["state_size"], params_passed=6, seed = seed)
    Drl = ReplayMemoryFast(memory_size=30000,
                           state_size=configuration["ModelBase"]["state_size"], params_passed=6, seed = seed)
    Eval_buffer = ReplayMemoryFast(memory_size=10000,
                                   state_size=configuration["ModelBase"]["state_size"], params_passed=6, seed = seed)

    seed = configuration["seed"]
    '''Initial training from random collected data'''
    filename = "Cheetah_withx.npy"
    data = np.load(filename, allow_pickle=True)
    np.random.shuffle(data)
    if configuration["run_settings"]["reduced_states"]:
        data = reduce_state(data)
    Drand.store(state = data[1:-3000,0], action = data[1:-3000,1], next_state = data[1:-3000,2], dones = data[1:-3000,3], rewards =  data[1:-3000,4])
    States, Actions, Next_States, _, Rewards = Drand.sample(replace = False)
    Outputs = Next_States - States
    mean_state, std_state = get_scalers(States)
    #np.save("State_scalar", np.vstack([mean_state,std_state]))
    mean_action, std_action = get_scalers(Actions)
    #np.save("Action_scalar", np.vstack([mean_action, std_action]))
    #mean_action, std_action = np.array([0.0,0.0,0.0,0.0]), np.array([1.0,1.0,1.0,1.0])
    mean_out, std_out = get_scalers(Outputs)
    #np.save("Output_scalar", np.vstack([mean_out, std_out]))
    mb_c = configuration["ModelBase"]
    save_prefix = configuration['save_prefix']
    DynamicModel = ModelBase(state_size = mb_c["state_size"], action_size = mb_c["action_size"],project_title = configuration["title"], state_mean = mean_state, state_std = std_state, action_mean = mean_action, action_std = std_action,
                             output_mean = mean_out, output_std = std_out,hidden_layers = mb_c["mb_layers"],
                             hidden_units = mb_c["networkUnits"], activation_d = mb_c["activation_d"], activation_op = mb_c["activation_op"],
                             batch_size = mb_c["mb_batchSize"], logger = logger_neptune, save_prefix = save_prefix)
    env_name = 'HalfCheetah-v3'
    ppo_c = configuration["PPO_settings"]
    agent = PPO_Agent(ppo_c, env_name, seed=seed, max_trajectories=configuration['MPC']['max_trajectories'], short_horizon=configuration['MPC']['horizon'],
                      title=title, save_prefix = save_prefix, logger_neptune = logger_neptune,
                      last_save= configuration["model_loader"]["load_after_iters"],
                      MPC_only = configuration["model_loader"]["MPC_only"],
                      reduced_obs = configuration["run_settings"]["reduced_states"], mpc_rand = configuration["run_settings"]["MPC_rand"])
    plotter = Plotter(title, logger_neptune, save_prefix)

    States = scale_data(mean_state, std_state, States)
    Actions = scale_data(mean_action, std_action, Actions)
    Outputs = scale_data(mean_out, std_out, Outputs)
    Outputs = add_noise(Outputs, 0.01)
    Inputs = np.concatenate([States, Actions], axis = 1 )
    Inputs = add_noise(Inputs, 0.01)

    '''Prepare Evaluation Data'''
    Eval_buffer.store(state = data[-3000:,0], action = data[-3000:,1], next_state = data[-3000:,2], dones = data[-3000:,3], rewards = data[-3000:, 4])
    Eval_buffer.save_to_file("Eval.npy")
    States_eval, Actions_eval, Next_States_eval, _, Rewards_eval = Eval_buffer.sample(replace = False)
    Outputs_eval = Next_States_eval - States_eval
    States_eval = scale_data(mean_state, std_state, States_eval)
    Actions_eval = scale_data(mean_action, std_action, Actions_eval)
    Outputs_eval = scale_data(mean_out, std_out, Outputs_eval)
    Inputs_eval = np.concatenate([States_eval, Actions_eval], axis = 1 )
    horizon = 100
    n = 10
    aggregation_counter = 0
    if configuration["run_settings"]["mode"] == "test":
        agent.test(configuration["model_loader"]["load_after_iters"])
    if configuration["model_loader"]["mb_load_model"]:
        DynamicModel.restore(configuration["model_loader"]["load_iters_mb"])
    else:
        DynamicModel.train(Inputs, Outputs, np.array([]), np.array([]), Rewards, Inputs_eval, Outputs_eval, Rewards_eval, epochs_dyn = mb_c['mb_init_epoch'],
                       epochs_rew = mb_c['rew_init_epoch'], aggregate_step = 0, fraction_use_new=0)
        if not os.path.isdir(os.path.join(save_prefix,"Iter{}".format(aggregation_counter))):
            os.mkdir(os.path.join(save_prefix,"Iter{}".format(aggregation_counter)))
        model_evaluate(1, horizon, agent, DynamicModel, plotter, aggregation_counter)
        model_evaluate(10, horizon, agent, DynamicModel, plotter, aggregation_counter)
        aggregation_counter +=1
    if configuration["model_loader"]["restore_model_from_file"]:
        agent.load(configuration["model_loader"]["load_after_iters"])
        aggregation_counter = 1
        '''Model evaluate 1 step'''
        model_evaluate(1, horizon, agent, DynamicModel, plotter, aggregation_counter )
        # '''Model evaluate n steps'''
        # model_evaluate(n, horizon, agent, DynamicModel, plotter, aggregation_counter )
        # '''Model evaluate full traj'''
        # model_evaluate(horizon, horizon, agent, DynamicModel, plotter, aggregation_counter )
        # '''Testing policy on real environement'''
        # agent.test(configuration["model_loader"]["load_after_iters"])
    if configuration["model_loader"]["MPC_only"]:
        runMBMF(agent, DynamicModel)
        return
    number_of_aggregates = mb_c["number_of_aggregates"]
    aggregate_every_iter = mb_c['aggregate_every_iter']
    fraction_use_new = mb_c['fraction_use_new']
    eval_ratio = 0.2
    # horizon for plotting
    epochs_dyn  = mb_c['mb_epoch']
    epochs_rew = mb_c['rew_epochs']
    while aggregation_counter <= number_of_aggregates:
        if not os.path.isdir(os.path.join(save_prefix,"Iter{}".format(aggregation_counter))):
            os.mkdir(os.path.join(save_prefix,"Iter{}".format(aggregation_counter)))
        data = agent.train(aggregate_every_iter, aggregation_counter,DynamicModel)
        np.random.shuffle(data)
        eval_index = int(data.shape[0]*eval_ratio)
        Drl.store(state=data[1:-eval_index, 0], action=data[1:-eval_index, 1], next_state=data[1:-eval_index, 2],
                    dones=data[1:-eval_index, 3], rewards=data[1:-eval_index, 4])
        Drl.save_to_file("Drl.npy")
        # States_rand, Actions_rand, Next_States_rand, _, Rewards_rand = Drand.sample()
        # Outputs_rand = Next_States_rand - States_rand
        # mean_state, std_state = get_scalers(States)
        # mean_action, std_action = get_scalers(Actions)
        # mean_out, std_out = get_scalers(Outputs)
        # States_rand = scale_data(mean_state, std_state, States_rand)
        # Actions_rand = scale_data(mean_action, std_action, Actions_rand)
        # Outputs_rand = scale_data(mean_out, std_out, Outputs_rand)
        # Inputs_rand = np.concatenate([States_rand, Actions_rand], axis=1)
        #
        # States_rl, Actions_rl, Next_States_rl, _, Rewards_rl = Drl.sample()
        # Outputs_rl = Next_States_rl - States_rl
        # States_rl = scale_data(mean_state, std_state, States_rl)
        # Actions_rl = scale_data(mean_action, std_action, Actions_rl)
        # Outputs_rl = scale_data(mean_out, std_out, Outputs_rl)
        # Inputs_rl = np.concatenate([States_rl, Actions_rl], axis=1)
        #
        # #Inputs = np.concatenate([Inputs_rand,Inputs_rl], axis = 0)
        # #Outputs = np.concatenate([Outputs_rand, Outputs_rl], axis = 0)
        # Rewards = np.concatenate([Rewards_rand.flatten(),Rewards_rl], axis = 0)
        '''Prepare evaluation data'''
        # Eval_buffer.store(state=data[-eval_index:, 0], action=data[-eval_index:, 1], next_state=data[-eval_index:, 2],
        #                   dones=data[-eval_index:, 3], rewards=data[-eval_index:, 4])
        # Eval_buffer.save_to_file("Eval.npy")
        # States_eval, Actions_eval, Next_States_eval, _, Rewards_eval = Eval_buffer.sample()
        # Outputs_eval = Next_States_eval - States_eval
        # States_eval = scale_data(mean_state, std_state, States_eval)
        # Actions_eval = scale_data(mean_action, std_action, Actions_eval)
        # Outputs_eval = scale_data(mean_out, std_out, Outputs_eval)
        # Inputs_eval = np.concatenate([States_eval, Actions_eval], axis=1)
        #Outputs = add_noise(Outputs, 0.001)
        # DynamicModel.train(Inputs_rand, Outputs_rand, Inputs_rl, Outputs_rl, Rewards, Inputs_eval, Outputs_eval, Rewards_eval, epochs_dyn = epochs_dyn,
        #                    epochs_rew = epochs_rew, aggregate_step = int(aggregation_counter), fraction_use_new = fraction_use_new)
        '''Model evaluate 1 step'''
        model_evaluate(1, horizon, agent, DynamicModel, plotter, aggregation_counter)
        '''Model evaluate n steps'''
        model_evaluate(n, horizon, agent, DynamicModel, plotter, aggregation_counter)
        '''Model evaluate full traj'''
        model_evaluate(horizon, horizon, agent, DynamicModel, plotter, aggregation_counter)
        '''Testing policy on real environement'''
        #agent.test(test_episodes=(aggregation_counter+1)*2,)
        aggregation_counter += 1
        Drl.clear()

def runMBMF(agent, DynamicModel):
    agent.train(model = DynamicModel)

def get_scalers(Data):
    mean = np.mean(Data, axis = 0)
    std = np.std(Data-mean, axis = 0)
    return mean, std

def scale_data_minmax(min, max, Data):
    return (Data - min)/(max-min)


def scale_data(mean, std, Data):
    transformed_data = (Data - mean)/std
    return transformed_data

def add_noise(data, noiseToSignal):
    data_mod = copy.deepcopy(data)
    mean_data = np.mean(data, axis=0)
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        if (std_of_noise[j] > 0):
            data_mod[:, j] = np.copy(data[:, j] + np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data_mod

def model_evaluate(nsteps, horizon, agent, model, plotter, iteration):
    mode = str(nsteps)+ "steps"
    test_env = agent
    test_model = model
    ob_env = test_env.env.reset()
    state_size = test_env.env.observation_space.shape[0]
    action_size = test_env.env.action_space.shape[0]
    ob_model = model.reset(ob_env)
    States_env = [ob_env for _ in range(horizon)]
    Reward_env = [0 for _ in range(horizon)]
    States_model = [ob_model for _ in range(horizon)]
    Reward_model = [0 for _ in range(horizon)]
    Dones_env = [0 for _ in range(horizon)]
    Dones_model = [0 for _ in range(horizon)]
    for h in range(1, horizon):
        if h%nsteps == 0:
            ob_model = test_model.reset(ob_env)
        _,ac_env, _ = test_env.actor_critic.act(ob_env)
        ob_env, rew, done_env, _ = test_env.env.step(ac_env)
        States_env[h] = ob_env
        Reward_env[h] = rew
        Dones_env[h] = done_env
        _,ac_model,_ = test_env.actor_critic.act(ob_model)
        ob_model, rew_model, done_model, _ = test_model.step(action = ac_env)
        States_model[h] = ob_model
        Reward_model[h] = rew_model
        Dones_model[h] = done_model
        if done_env:
            ob_env = test_env.env.reset()
            ob_model = test_model.reset(ob_env)

    plotter.plot_states(States_env, States_model, iteration, mode)
    plotter.plot_rewards(Reward_env, Reward_model, iteration, mode)
    #plotter.plot_done(Dones_env, Dones_model, iteration, mode)

def get_parameters(configs):
    return configs

if __name__ == '__main__':

    global title
    title = sys.argv[1]
    c = Variables(title=title)

    print("-------------------------------")
    print("Description of experiment:")
    print("-------------------------------")
    print(c.get_data()['description'])
    print("-------------------------------")
    input('press enter to continue')

    configuration = c.get_data()

    if configuration["run_settings"]["logger"]:
        logger_neptune = neptune.init(
            project="anjalinair012/Cheetah-MB",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZWFjNmViZi00NmE5LTQzZjktYjcwNi03ZjU0MjBlY2M2NmEifQ==",
            capture_stdout=False,
            capture_stderr=False,
        )
        params = get_parameters(configuration)
        logger_neptune["parameters"] = params
    else:
        logger_neptune = None

    train(configuration, logger_neptune)


