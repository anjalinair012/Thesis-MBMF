import os

import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, title, logger, save_prefix):
        self.title = title
        self.logger = logger
        self.path = save_prefix


    def plot_rewards(self, rew_real, rew_predicted, header="", iteration = 0, mode="1step"):
        fig = plt.figure()
        plt.plot(rew_real)
        plt.plot(rew_predicted)
        plt.title(header)
        if self.logger:
            self.logger["{}/{}/Rew_compared".format(iteration, mode)].upload(fig)
        else:
            plt.savefig(os.path.join(self.path, "Rew_compared_{}_{}.png".format(str(iteration), mode)))
        plt.close()

    def plot_done(self, done_real, done_predicted, header="", iteration = 0, mode = "1step"):
        fig = plt.figure()
        plt.plot(done_real)
        plt.plot(done_predicted)
        plt.title(header)
        if self.logger:
            self.logger["{}/{}/Term_compared".format(iteration, mode)].upload(fig)
        else:
            plt.savefig(os.path.join(self.path, "\{}Term_compared_{}_{}.png".format(str(iteration), mode)))
        plt.close()


    def plot_states(self, real_states, predicted_states, iteration, mode = "1step"):
        state_size = real_states[0].shape[0]
        state_labels = ['xpos', 'zfront', 'angleFront', 'angleSecondRot', 'velxfront', 'velyfront', 'omegaSecondRot', 'xfront',
                        'yfront', 'omegafront', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16']
        fig = plt.figure()
        for o in range(state_size):
            data_pred = []
            data_real = []
            for i in range(len(predicted_states)):
                data_real.append(real_states[i][o])
                data_pred.append(predicted_states[i][o])
            plt.plot(data_real[:100])
            plt.plot(data_pred[:100])
            plt.title(state_labels[o])
            plt.legend(['real', 'predicted'], loc='upper left')
            if self.logger:
                self.logger["{}/{}/{}".format(iteration, mode, state_labels[o])].upload(fig)
            plt.savefig(os.path.join(self.path, "Iter{}".format(iteration), '{}_{}'.format(state_labels[o], mode)))
            plt.clf()