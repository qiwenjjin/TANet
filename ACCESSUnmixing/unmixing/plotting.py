from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.models import Model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy.io as sio


def numpy_SAD(y_true, y_pred):
    return np.arccos(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))


def compute_ASAM_rad(endmembers, endmembersGT, dict):
    num_endmembers = endmembers.shape[0]
    if dict is None:
        dict = order_endmembers(endmembers, endmembersGT)

    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        # ASAM=ASAM+numpy_SAD(endmembers[i,:],endmembersGT[dict[i],:]) #endmembers[i,:].dot(endmembersGT[dict[i]])/(np.linalg.norm(endmembers[i,:])*np.linalg.norm(endmembersGT[dict[i],:]))
        # ASAM = ASAM+np.arccos(endmembers[i, :].dot(endmembersGT[dict[i]]) / (
        #     np.linalg.norm(endmembers[i, :]) * np.linalg.norm(endmembersGT[dict[i], :])))
        if np.var(endmembersGT[dict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[dict[i]])
            num += 1
    return ASAM / float(num)


def plotEndmembersAndGT(endmembersGT, endmembers, dict, normalize=True, figure_nr=11):
    num_endmembers = endmembers.shape[0]
    n = num_endmembers / 2  # how many digits we will display
    if num_endmembers % 2 != 0: n = n + 1
    if dict is None:
        dict = order_endmembers(endmembers, endmembersGT)
    aSAM = compute_ASAM_rad(endmembers, endmembersGT, dict)
    fig = plt.figure(num=figure_nr, figsize=(14, 8))
    plt.clf()
    title = "aSAM score for all endmembers: " + format(aSAM, '.3f') + " radians"
    st = plt.suptitle(title)
    if normalize:
        for i in range(num_endmembers):
            endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
            endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, :], 'r', linewidth=1.0)
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembersGT[dict[i], :], 'k', linewidth=1.0)
        ax.set_title("SAD: " + str(i) + " :" + format(numpy_SAD(endmembers[i, :], endmembersGT[dict[i], :]), '.4f'))
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)

    plt.draw()
    plt.pause(0.001)
    return dict


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in dict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in dict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            dict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1

    return dict


def plotAbundancesSimple(num_endmembers, size_data, abundances, dict, use_ASC=1, figure_nr=16):
    # abundances = np.squeeze(abundances)
    abundances = np.reshape(abundances, (size_data[1], size_data[0], num_endmembers))
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    # if dict is not None and is_GT==False:
    #     # scores = compute_GSME(abundances, abundancesGT, dict)
    # else:
    n = num_endmembers / 2
    if num_endmembers % 2 != 0: n = n + 1
    fig = plt.figure(num=figure_nr, figsize=(14, 8))
    # AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        # NormedAbundances = abundances[:, :, i] / AA
        # if use_ASC:
        # im = ax.imshow(abundances[:, :, i] / np.max(abundances[:, :, i]), cmap='jet')
        im = ax.imshow(abundances[:, :, i], cmap='viridis')
        # im = ax.imshow(abundances[:, :, i], cmap='jet')
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im.set_clim([0, 1])

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def plotEndmembers(num_endmembers, endmembers, normalize=False, figure_nr=11):
    n = num_endmembers / 2  # how many digits we will display
    if num_endmembers % 2 != 0: n = n + 1
    plt.figure(num=figure_nr, figsize=(14, 8))
    plt.clf()
    if normalize:
        endmembers = endmembers / endmembers.max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, :], 'r', linewidth=1.0)

        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


class PlotWhileTraining(Callback):
    def __init__(self, plot_every_n, size, num_endmembers, data, endmembersGT, plot_GT, plot_S, num_inputs=1):
        super(PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.num_endmembers = num_endmembers
        self.input = data
        self.endmembersGT = endmembersGT
        self.plotGT = plot_GT
        self.plotS = plot_S
        self.num_epochs = 0
        self.losses = []
        self.sads = []
        self.size = size
        self.num_inputs = num_inputs
        self.mse = []

    def on_train_begin(self, logs={}):
        self.losses = []
        self.num_epochs = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        # if batch%10==0:
        #     endmembers = self.model.get_layer('endmembers').get_weights()[0]
        #     if self.plotGT:
        #         sad = compute_ASAM_rad(endmembers, self.endmembersGT, None)
        #         self.sads.append(sad)
        #         # spio.savemat('sad.mat', {'SAD': self.sads,'M':endmembers,'M_GT':self.endmembersGT})
        # if batch%50==0:
        #     spio.savemat('sad.mat', {'SAD': self.sads, 'LOSS': self.losses})
        return

    def on_epoch_end(self, epoch, logs=None):
        if self.endmembersGT is None:
            self.plotGT = False
        self.losses.append(logs.get('loss'))
        self.num_epochs = epoch
        # self.losses.append(logs.get('loss'))
        endmembers = self.model.get_layer('endmembers').get_weights()[0]
        if self.plotGT:
            sad = compute_ASAM_rad(endmembers, self.endmembersGT, None)
            self.sads.append(sad)
            # print('SAD: ' + str(sad))
            # spio.savemat('sad.mat', {'SAD': self.sads,'M':endmembers,'M_GT':self.endmembersGT})
            sio.savemat('sad.mat', {'SAD': self.sads, 'LOSS': self.losses})
        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0: return
        if self.plotS:
            if self.num_inputs > 1:
                intermediate_layer_model = Model(inputs=self.model.input,
                                                 outputs=[self.model.get_layer('abundances' + str(i)).output for i in
                                                          range(self.num_inputs)])
                abundances = np.mean(
                    intermediate_layer_model.predict([self.input.orig_data for i in range(self.num_inputs)]),
                    axis=0)
            else:
                intermediate_layer_model = Model(inputs=self.model.input,
                                                 outputs=self.model.get_layer('abundances').output)
                abundances = intermediate_layer_model.predict(self.input.orig_data)
            if self.size is None:
                self.size = (int(np.sqrt(abundances.shape[0])), int(np.sqrt(abundances.shape[0])))

        # plotHist(self.losses, 33)
        # self.plotGT = False
        if self.plotGT:
            dict = order_endmembers(endmembers, self.endmembersGT)
            # if self.is_GT_for_A:
            #     plotAbundances(self.num_endmembers, self.size_data, abundances, self.abundancesGT, dict, self.use_ASC)
            # else:
            #     plotAbundances(self.num_endmembers, self.size_data, abundances, None, None, self.use_ASC, is_GT=False)
            plotEndmembersAndGT(self.endmembersGT, endmembers, dict)
            if self.plotS:
                plotAbundancesSimple(self.num_endmembers, (self.size[0], self.size[1]), abundances, dict,
                                     use_ASC=1, figure_nr=10)
        else:
            # plotAbundances(self.num_endmembers, self.size_data, abundances, None, None, self.use_ASC)
            plotEndmembers(self.num_endmembers, endmembers)
            plotAbundancesSimple(self.num_endmembers, (self.size[0], self.size[1]), abundances, None,
                                 use_ASC=1, figure_nr=10)
        return
