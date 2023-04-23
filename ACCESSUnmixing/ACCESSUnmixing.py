import os
import sys
import scipy.io as sio
import numpy as np
import random
from tensorflow.python.keras.constraints import non_neg
from tensorflow.python.keras.layers import LeakyReLU, Input, Dense, BatchNormalization, GaussianDropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers as optimizers
from tqdm import tqdm
from keras_tqdm import TQDMCallback
from tensorflow import set_random_seed

if __package__ == "ACCESSUnmixing":
    from ACCESSUnmixing.unmixing.HSI import HSI
    from ACCESSUnmixing.unmixing.losses import SAD
    from ACCESSUnmixing.unmixing.plotting import PlotWhileTraining
    from ACCESSUnmixing.frosti import utils
else:
    from unmixing.HSI import HSI
    from unmixing.losses import SAD
    from unmixing.plotting import PlotWhileTraining
    from frosti import utils

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
set_random_seed(random_seed)

# matplotlib.use('TkAgg')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Autoencoder:
    def __init__(self, n_end, data, activation=LeakyReLU(0.1), optimizer='adam', lr=0.001, l2=0.0, l1=0.00000,
                 is_GT=True, plot_every_n = 10):
        self.n_end = n_end
        self.hsi = data
        self.activation = activation
        self.lr = lr
        self.l2 = l2
        self.l1 = l1
        self.optimizer = optimizer
        # self.optimizer = optimizers.Adam(lr=self.lr)
        self.model = None
        self.use_bias = False
        self.abundance_layer = None
        self.initializer = initializers.glorot_normal()
        self.sum_to_one = True
        self.is_GT = is_GT
        self.plot_every_n = plot_every_n
        self.plotS = True
        self.weights = None
        self.is_deep = False
        self.activation = activation

    def create_model(self, loss):
        use_bias = False
        # Input layer
        input_ = Input(shape=(self.hsi.n_bands,))
        # Encoder
        if self.is_deep:
            encoded = Dense(self.n_end * 9, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=self.activation)(input_)
            # en coded = BatchNormalization()(encoded)
            encoded = Dense(self.n_end * 6, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=self.activation)(encoded)
            encoded = Dense(self.n_end * 3, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=self.activation)(encoded)
            encoded = Dense(self.n_end, use_bias=use_bias, kernel_regularizer=None, activity_regularizer=None,
                            activation=self.activation)(encoded)
        else:
            encoded = Dense(self.n_end, use_bias=use_bias, activation=self.activation, activity_regularizer=None,
                            kernel_regularizer=None)(input_)
        # Utility Layers

        # Batch Normalization
        encoded = BatchNormalization()(encoded)
        # Soft Thresholding
        encoded = utils.SparseReLU(alpha_initializer='zero', alpha_constraint=non_neg(), activity_regularizer=None)(
            encoded)
        # Sum To One (ASC)
        encoded = utils.SumToOne(axis=0, name='abundances', activity_regularizer=None)(encoded)

        decoded = GaussianDropout(0.0045)(encoded)

        # Decoder
        decoded = Dense(self.hsi.n_bands, activation='linear', name='endmembers', use_bias=use_bias,
                        kernel_constraint=non_neg(), kernel_regularizer=None, kernel_initializer=self.initializer)(
            encoded)
        self.model = Model(inputs=input_, outputs=decoded, name='Autoencoder')
        # Compile Model

        self.model.compile(self.optimizer, loss, metrics=[utils.SAD])

    # Fit Model
    def fit(self, epochs, batch_size):

        progress = TQDMCallback(leave_outer=True, leave_inner=True)
        setattr(progress, 'on_train_batch_begin', lambda x, y: None)
        setattr(progress, 'on_train_batch_end', lambda x, y: None)
        plotWhileTraining = PlotWhileTraining(self.plot_every_n, self.hsi.size, self.n_end, self.hsi, self.hsi.GT,
                                              self.is_GT, True)
        hist = self.model.fit(self.hsi.data, self.hsi.data, epochs=epochs, batch_size=batch_size, verbose=0,
                              callbacks=[progress, plotWhileTraining], shuffle=True)
        return hist

    # Shuffle or reset weights
    def shuffle_weights(self, weights):
        if weights is None:
            weights = self.model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        self.model.set_weights(weights)

    def get_endmembers(self):
        return self.model.layers[len(self.model.layers) - 1].get_weights()[0]

    def get_abundances(self):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer('abundances').output)
        abundances = intermediate_layer_model.predict(self.hsi.orig_data)
        return abundances

    def save_results(self, out_dir, fname):
        if out_dir is not None:
            out_path = out_dir / fname
        else:
            out_path = fname
        endmembers = self.get_endmembers()
        abundances = self.get_abundances()
        sio.savemat(out_path, {'M': endmembers, 'A': abundances})


if __name__ == '__main__':

    datadir = '../../Datasets'  # PATH TO DATASETS
    dataset = 'Samson'  # DATASET
    resdir = './Results/'  # Where to save results
    fname = datadir + '/' + dataset + '.mat'
    num_endmembers = 3
    nx = 100
    myData = HSI(fname)
    if len(sys.argv) > 1:
        number_of_runs = sys.argv[1]
    else:
        number_of_runs = 1

    for i in tqdm(range(number_of_runs)):
        myData = HSI(fname)
        myData.load_data(normalize=True, shuffle=False)
        unmixer = Autoencoder(n_end=num_endmembers, data=myData, activation=LeakyReLU(0.2),
                              optimizer=optimizers.RMSprop(0.001), l2=0.0000, l1=0.00000,
                              plot_every_n = 0)
        unmixer.create_model(SAD)
        myData.make_patches(1, num_patches=2000)
        unmixer.fit(epochs=100, batch_size=7)
        resfile = dataset + '_run' + str(i) + '.mat'
        unmixer.save_results(resdir, resfile)
        del unmixer
        K.clear_session()
