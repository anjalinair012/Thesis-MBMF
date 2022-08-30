import yaml as yml
import os

run = None

class Variables:
    def __init__(self, title=''):
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        self.base_path = os.path.join(self.base_path, 'PPO-MUJoco-project-Results')
        if not os.path.isdir(self.base_path):
            os.mkdir(self.base_path)
        self.base_path = os.path.join(self.base_path, title)
        if not os.path.isdir(self.base_path):
            os.mkdir(self.base_path)
        if not os.path.isdir(os.path.join(self.base_path, "Models")):
            os.mkdir(os.path.join(self.base_path, "Models"))

        self.file_name = 'variables.yml'
        self.file_path = os.path.join(self.base_path, self.file_name)

        self.title = title

        print(f"\n\n{self.base_path}\n\n")

        if os.path.exists(self.file_path):
            print(f'variable values read from {self.file_path}')
            self.read_variables()
        else:
            print('variable file does not exist\ncreating new file using default configuration values')
            self.data = self.defaults()
            self.save_variables()
            self.wait_for_change()
            self.read_variables()

        #self.pc_specific_changes()

    # self.save_variables()
    # print(self.data)

    def defaults(self):
        data = {
            'title': self.title,
            'seed' : 0,
            'save_prefix' : self.base_path,
            'mode' : 'test',
            'description': '[Enter description of experiment]',
            'run_settings': {
                'workers': 1,
                'max_timesteps': 50000000,
                'logger': run,
                'reduced_states': False,
                'MPC_rand' : False
            },
            'PPO_settings' : {
                'NUM_ENV_TIMESTEPS' : 1e6,
                'NUM_TIMESTEPS_PER_UPDATE': 2048,
                'NUM_MINIBATCHES': 32,
                'NUM_EPOCHS': 4,

                'GAMMA': 0.99,
                'CLIP_PARAM': 0.2,
                'LAMBDA': 0.95,
                'VALUE_FUNCTION_COEF': 0.5,
                'ENTROPY_COEF': 0.0,

                'ACTOR_HIDDEN_UNITS': [64, 64],
                'CRITIC_HIDDEN_UNITS': [64, 64],
                'LEARNING_RATE': 3e-4,
                'OPTIMIZER_EPSILON': 1e-7,
                # Optimizer uses epsilon to avoid a divide by zero error when updating parameters
                # when the gradient is almost zero. Therefore it should be a small number, but not
                # too small since it can cause large weight updates.

                'SAVE_INTERVAL': 20,
                'PPO_Tuning' : False,
            },
            'model_loader': {
                'restore_model_from_file': False,
                'load_after_iters': 0,
                'mb_load_model': False,
                'load_iters_mb': -10,
                'MPC_only' : False
            },
            'ModelBase': {
                'state_size': 18,
                'action_size': 6,
                'activation_d': 'relu',
                'activation_op': 'linear',
                'mb_layers': 2,
                'mb_batchSize': 512,
                'networkUnits': 500,
                'load_buffers': False,
                'load_scalar': False,
                'mb_init_epoch': 60,
                'rew_init_epoch' : 50,
                'mb_epoch': 100,
                'rew_epochs' : 100,
                'aggregate_every_iter': 2,
                'number_of_aggregates' : 10,
                'fraction_use_new': 0.3,
            },
            'MPC': {
                'horizon': 10,
                'max_trajectories': 15,
            }
        }

        return data

    def get_data(self):
        return self.data

    def pc_specific_changes(self):
        self.data['ppo_settings']['save_prefix'] = self.base_path

    def wait_for_change(self):
        print(f'Default configuration save to {self.file_path}')
        print('to change configuration before run, edit and save this file')
        print('then press enter')
        input('\n waiting')

    def read_variables(self):
        with open(self.file_path) as f:
            self.data = yml.load(f, Loader=yml.FullLoader)  # yaml.load_all('data.yml')

    def save_variables(self):
        with open(self.file_path, 'w') as outfile:
            yml.dump(self.data, outfile, default_flow_style=False)