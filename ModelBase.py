import os
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import LeakyReLU, ReLU

class ModelBase:
    def __init__(self, state_size, action_size, project_title, state_mean, state_std, action_mean, action_std, output_mean, output_std,
                 hidden_layers, hidden_units, activation_d, activation_op, batch_size, logger, save_prefix, seed = 0):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.title = project_title
        self.lr = 0.001
        self.state_size = state_size
        self.action_size = action_size
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.output_mean = output_mean
        self.batch_size = batch_size
        self.output_std = output_std
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        #self.es_dyn = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # self.es_rew = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.dyn_model = self.create_model(hidden_layers, hidden_units, activation_d, activation_op, "Adam")
        # self.reward_function = self.Reward_function(1, 15, "linear", "linear", "Adam")
        self.state = None
        self.logger = logger
        self.save_prefix = save_prefix



    def create_model(self, hidden_layers = 1, hidden_units = 500, activation_d = "relu", activation_op = None, optimizer= "adam"):
        outputSize = self.state_size
        inputSize = self.state_size + self.action_size
        self.loss_object = keras.losses.MeanSquaredError() # trial
        self.optimizer = keras.optimizers.Adam(self.lr)
        initializer = tf.keras.initializers.GlorotNormal(seed=312)
        weights_reg = tf.keras.regularizers.l2(0.00)
        input_x = layers.Input(shape=(inputSize,))
        # hidden layer 1
        for layer in range(hidden_layers):

            if layer == 0:
                x1 = layers.Dense(hidden_units, kernel_regularizer=weights_reg, kernel_initializer=initializer,
                                   bias_initializer=initializer, activation = "relu")(input_x)
            else:
                x1 = layers.Dense(hidden_units, kernel_regularizer=weights_reg, kernel_initializer=initializer,
                                   bias_initializer=initializer, activation = "relu")(x1)
        # output layer
        output_x = layers.Dense(outputSize, kernel_regularizer=weights_reg, kernel_initializer=initializer,
                               bias_initializer=initializer)(x1)
        self.dyn_model = keras.Model(inputs = input_x, outputs = output_x)
        #model.compile(optimizer=optimizer, loss=loss_object, metrics=[test_loss, ])
        return self.dyn_model


    def Reward_function(self, hidden_layers, hidden_units, activation_d, activation_op, optimizer):
        outputSize = 1
        inputSize = 28
        loss_object = keras.losses.MeanSquaredError()
        test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
        optimizer = keras.optimizers.Adam(0.001)
        model = keras.Sequential()
        initializer = tf.keras.initializers.GlorotNormal(seed=312)
        model.add(layers.Input(shape=(inputSize,)))
        model.add(layers.Dense(15, kernel_initializer=initializer,
                               bias_initializer=initializer))
        # output layer
        model.add(layers.Dense(outputSize, kernel_initializer=initializer,
                               bias_initializer=initializer))
        model.compile(optimizer=optimizer, loss=loss_object, metrics=[test_loss, ])
        return model


    def restore(self, iteration):
        print("---restore model------")
        self.dyn_model.load_weights(self.save_prefix + '/Models/best_dyn{}.h5'.format(iteration))
        return

    def train(self,Inputs_old, Outputs_old, Inputs_new, Outputs_new, rewards, Inputs_eval, Outputs_eval,
              rewards_eval, epochs_dyn, epochs_rew, aggregate_step, fraction_use_new):
        self.dyn_model = self.create_model(self.hidden_layers, self.hidden_units)

        train_loss_metric = tf.keras.metrics.MeanSquaredError(name='train_loss')
        val_loss_metric = tf.keras.metrics.MeanSquaredError(name='test_loss')

        training_loss_list = []
        validation_loss_list = []
        nData_old = Inputs_old.shape[0]
        num_new_pts = Inputs_new.shape[0]
        range_of_indeces = np.arange(Inputs_old.shape[0])
        #how much of new data to use per batch
        if(num_new_pts<(self.batch_size*fraction_use_new)):
            batchsize_new_pts = num_new_pts #use all of the new ones
        else:
            batchsize_new_pts = int(self.batch_size*fraction_use_new)

        #how much of old data to use per batch
        batchsize_old_pts = int(self.batch_size- batchsize_new_pts)
        for i in range(epochs_dyn):

            # reset to 0
            avg_loss = 0
            num_batches = 0
            # train from both old and new dataset
            if (batchsize_old_pts > 0):

                # get through the full old dataset
                # randomly order indeces (equivalent to shuffling dataX and dataZ)
                old_indeces = np.random.choice(range_of_indeces, size=(Inputs_old.shape[0],), replace=False)
                for batch in range(int(nData_old / batchsize_old_pts)):

                    # randomly sample points from new dataset
                    if (num_new_pts == 0):
                        dataX_new_batch = Inputs_new
                        dataZ_new_batch = Outputs_new
                    else:
                        new_indeces = np.random.randint(0, Inputs_new.shape[0], (batchsize_new_pts,))
                        dataX_new_batch = Inputs_new[new_indeces, :]
                        dataZ_new_batch = Outputs_new[new_indeces, :]

                    # walk through the randomly reordered "old data"
                    dataX_old_batch = Inputs_old[old_indeces[batch * batchsize_old_pts:(batch + 1) * batchsize_old_pts], :]
                    dataZ_old_batch = Outputs_old[old_indeces[batch * batchsize_old_pts:(batch + 1) * batchsize_old_pts], :]

                    if batchsize_new_pts>0:
                        # combine the old and new data
                        dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
                        dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))
                    else:
                        dataX_batch = dataX_old_batch
                        dataZ_batch = dataZ_old_batch

                    with tf.GradientTape() as tape:
                        logits = self.dyn_model(dataX_batch, training=True)
                        loss_value = self.loss_object(dataZ_batch, logits)
                    grads = tape.gradient(loss_value, self.dyn_model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.dyn_model.trainable_weights))

                    train_loss_metric.update_state(dataZ_batch, logits)
                    if batch % 200 == 0:
                        print(
                            "Training loss (for one batch) at step %d: %.4f"
                            % (batch, float(loss_value))
                        )
                        print("Seen so far: %s samples" % ((batch + 1) * self.batch_size))
            train_loss = train_loss_metric.result()
            for batch in range(int(Inputs_eval.shape[0] / self.batch_size)):
                val_indeces = np.random.randint(0, Inputs_eval.shape[0], (self.batch_size,))
                Eval_batch_Inputs = Inputs_eval[val_indeces, :]
                Eval_batch_Outputs = Outputs_eval[val_indeces, :]
                val_logits = self.dyn_model(Eval_batch_Inputs, training=False)
                # Update val metrics
                val_loss_metric.update_state(Eval_batch_Outputs, val_logits)
            val_loss = val_loss_metric.result()
            print("Epoch : {}----train_loss: {}------val_loss: {}--------".format(i,str(train_loss.numpy()), str(val_loss.numpy())))
            training_loss_list.append(train_loss)
            validation_loss_list.append(val_loss)
            val_loss_metric.reset_states()
            train_loss_metric.reset_states()
            num_batches += 1
            # else:
            #     for batch in range(int((num_new_pts / batchsize_new_pts))):
            #         # walk through the shuffled new data
            #         dataX_batch = Inputs_new[batch * batchsize_new_pts:(batch + 1) * batchsize_new_pts, :]
            #         dataZ_batch = Inputs_new[batch * batchsize_new_pts:(batch + 1) * batchsize_new_pts, :]
            #
            #         # one iteration of feedforward training
            #         _, loss, output, true_output = self.sess.run(
            #             [self.train_step, self.mse_, self.curr_nn_output, self.z_],
            #             feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            #
            #         training_loss_list.append(loss)
            #         avg_loss += loss
            #         num_batches += 1

                # shuffle new dataset after an epoch (if training only on it)
                # p = np.random.permutation(Inputs_new.shape[0])
                # dataX_new = Inputs_new[p]
                # dataZ_new = Outputs_new[p]

        plt.plot(training_loss_list)
        plt.plot(validation_loss_list)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.save_prefix + "/train_plot_dyn{}.png".format(aggregate_step))
        f = open(self.save_prefix + "/Model_loss.txt", "a+")
        f.write("Train loss {}  ".format(str(training_loss_list[-1])))
        f.write("Validation loss {}\r\n   ".format(str(validation_loss_list[-1])))
        f.close()
        if self.logger:
            self.logger["Model_loss"].log(validation_loss_list[-1])
        m = open(self.save_prefix + "/Model_config.txt", "a+")
        m.write(str(self.optimizer.get_config()))
        m.close()
        self.dyn_model.save_weights(self.save_prefix+'/Models/best_dyn{}.h5'.format(aggregate_step))
        #self.dyn_model.load_weights(self.save_prefix+'/Models/best_dyn.h5')
        # mc_rew = ModelCheckpoint(self.save_prefix+'/best_rew{}'.format(aggregate_step), monitor='val_loss', mode='min', verbose=1,
        #              save_best_only=True,
        #              save_weights_only=True)
        # history_rew = self.reward_function.fit(Inputs, rewards, epochs = epochs_rew, batch_size=self.batch_size, steps_per_epoch=200,
        #                                  validation_data=(Inputs_eval, rewards_eval), validation_steps=2,
        #                                  callbacks=[self.es_rew,mc_rew])
        # plt.plot(history_rew.history['loss'])
        # plt.plot(history_rew.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig(self.save_prefix + "/train_plot_rew{}.png".format(aggregate_step))


    def reset(self, state):
        self.state = state
        self.prev_state = state
        return self.state


    def predict(self, Input):
        output_dyn = self.dyn_model.predict(Input)
        #output_rew = self.reward_function.predict(Input)
        return output_dyn, None

    def step(self, state = np.array([]), action = None):
        self.prev_state = self.state
        action = np.clip(action, a_min=-1, a_max=1)
        if not np.any(state):
            state = self.state
        state_mod = (state - self.state_mean)/self.state_std
        action_mode = (action - self.action_mean)/self.action_std
        Input = np.reshape(np.concatenate([state_mod,action_mode], axis=0), [1,-1])
        nxt_state = self.dyn_model.predict(Input)
        #nxt_state, _ = self.predict(Input)
        self.state = np.reshape(nxt_state*self.output_std + self.output_mean + state, [self.state.shape[0]])
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (self.state[0] - self.prev_state[0])*200
        reward = reward_ctrl + reward_run
        return  self.state, reward, None, None

    def multi_predict(self, states, actions):
        states_mod = (states - self.state_mean)/self.state_std
        actions_mod = (actions - self.action_mean)/self.action_std
        Inputs = np.concatenate([states_mod, actions_mod], axis=1)
        #Reward Original
        # rewards_ctrl = -0.1 * np.sum(np.square(actions), axis = 1)
        # reward_run = (Next_States[:,0] - states[:,0]) * 200
        return self.dyn_model.predict(Inputs, verbose = 0) * self.output_std + self.output_mean + states


    def Scores(self, States, Next_States, Actions):
        rewards_ctrl = -0.1 * np.sum(np.sum(np.square(Actions), axis = 0), axis = 1)
        Rewards = np.sum((Next_States[:,:,0] - States[:,:,0]), axis = 0) + rewards_ctrl

        #Reward from
        # heading_penalty_factor = 10
        # Rewards = np.zeros((States.shape[0],States.shape[1]))
        #
        # #dont move front shin back so far that you tilt forward
        # front_leg = States[:, :, 5]
        # my_range = 0.2
        # Rewards[front_leg >= my_range] += heading_penalty_factor
        #
        # front_shin = States[:, :, 6]
        # my_range = 0
        # Rewards[front_shin >= my_range] += heading_penalty_factor
        #
        # front_foot = States[:, :, 7]
        # my_range = 0
        # Rewards[front_foot >= my_range] += heading_penalty_factor
        #
        # Rewards -= (Next_States[:,:, 17] - States[:, :, 17]) / 0.01
        # return np.sum(Rewards, axis = 0)
        return Rewards