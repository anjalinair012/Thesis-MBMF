import numpy as np

from ModelBase import ModelBase

state_size=18
action_size = 6
project_title=""
state_scalars = np.load("State_scalar.npy")
action_scalars = np.load("Action_scalar.npy")
output_scalars = np.load("Output_scalar.npy")
model_obj = ModelBase(state_size, action_size, project_title, state_scalars[0], state_scalars[1], action_scalars[0],
                      action_scalars[1], output_scalars[0], output_scalars[1],
                 2, 500, "relu", "linear", 526, None, "", seed = 0)

model_obj.dyn_model.load_weights("best_dyn0.h5")
