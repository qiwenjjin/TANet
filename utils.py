import tensorflow.contrib.keras.api.keras
from collections import OrderedDict
import matplotlib
from tensorflow.contrib.keras.api.keras import regularizers, initializers, constraints, losses
from tensorflow.contrib.keras.api.keras.constraints import Constraint
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Layer
import numpy as np
from tensorflow.contrib.keras.api.keras import initializers, regularizers, activations, constraints, optimizers

from tensorflow.contrib.keras.api.keras.regularizers import Regularizer
from tensorflow.contrib.keras.api.keras.utils import deserialize_keras_object, serialize_keras_object
import tensorflow.contrib.keras.api.keras.backend as K
import six
from tensorflow.contrib.keras.api.keras.layers import Layer
from tensorflow.contrib.keras.api.keras.layers import InputSpec
from tensorflow.contrib.keras.api.keras import optimizers as optimizers
import tensorflow.contrib.keras.api.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from tensorflow.contrib.keras.api.keras import callbacks
import tensorflow as T
from tensorflow.contrib.keras.api.keras.callbacks import Callback
# import theano.tensor as th
import scipy.io as sio
from scipy.fftpack import dct, idct

"""Various custom layers and utility functions
SparseReLu layer with dynamic thresholding
ASC SumToOne layer
"""
from tensorflow.contrib.keras.api.keras.regularizers import Regularizer


# Dynamic thresholding ReLU layer
class SparseReLU(Layer):
    def __init__(self, alpha_initializer='zeros',
                 activity_regularizer=None,
                 alpha_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(SparseReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self.alpha = self.add_weight(shape=input_shape[1:], name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.activity_regularizer,
                                     constraint=self.alpha_constraint)
        super(SparseReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.relu(x - self.alpha)
        # return K.relu(x - self.alpha)

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint)
        }
        base_config = super(SparseReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseLeakyReLU(Layer):
    def __init__(self, alpha_initializer='zeros',
                 activity_regularizer=None,
                 alpha_constraint=None,
                 slope_constraint=None,
                 slope=0.3,
                 shared_axes=None,
                 **kwargs):
        super(SparseLeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.slope_initializer = initializers.constant(slope)
        self.slope_constraint = constraints.get(slope_constraint)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self.alpha = self.add_weight(input_shape[1:],
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     # regularizer=self.activity_regularizer,
                                     constraint=self.alpha_constraint)
        self.slope = self.add_weight(input_shape[1:],
                                     name='slope',
                                     initializer=self.slope_initializer,
                                     # regularizer=self.activity_regularizer,
                                     constraint=self.slope_constraint)
        super(SparseLeakyReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.relu(x - self.alpha, alpha=self.slope)

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint)
        }
        base_config = super(SparseLeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SumToOne(Layer):
    def __init__(self, axis=0, activity_regularizer=None, **kwargs):
        self.axis = axis
        self.uses_learning_phase = True
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(SumToOne, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # return K.softmax(3*x)
        x *= K.cast(x >= 0., K.floatx())
        # x = K.abs(x)
        # x = x - K.min(K.flatten(x))
        x = K.transpose(x)
        x_normalized = x / (K.sum(x, axis=0) + K.epsilon())
        x = K.transpose(x_normalized)
        # x *= K.cast(x/K.max(x) >= 0.3, K.floatx())
        # x = 4.0*x/(K.max(K.flatten(x))+K.epsilon())
        # x=K.softmax(x)
        return x

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SumToOne, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

        # Custom nonnegative unit norm constraint


class NonNegUnitNorm(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        p = K.switch(K.min(p) < 0, p - K.min(p) + K.epsilon(), p + K.epsilon())
        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}


class NonNegUnitNorm2(Constraint):
    '''Constrain the weights to be non-negative.
         '''

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p / (K.epsilon() + K.sqrt(K.sum(K.square(p), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}


class PlotafterEpoch2(Callback):
    def __init__(self, plot_every_n, size, num_endmembers, data, endmembersGT, plot_GT, plot_S):
        super(PlotafterEpoch2, self).__init__()
        self.plot_every_n = plot_every_n
        self.num_endmembers = num_endmembers
        self.input = data
        self.endmembersGT = endmembersGT
        self.plotGT = plot_GT
        self.plotS = plot_S
        self.num_epochs = 0
        self.losses = []
        self.size = size

    def on_train_begin(self, logs={}):
        self.losses = []
        self.num_epochs = 0

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.num_epochs = epoch
        self.losses.append(logs.get('loss'))
        if epoch % self.plot_every_n != 0: return
        return
        if self.plotS:
            intermediate_layer_model = Model(inputs=self.model.input,
                                             outputs=self.model.get_layer('abundances').output)
            abundances = intermediate_layer_model.predict(self.input)
            if self.size is None:
                self.size = (int(np.sqrt(abundances.shape[0])), int(np.sqrt(abundances.shape[0])))
        endmembers = self.model.layers[len(self.model.layers) - 1].get_weights()[0]

        # plotHist(self.losses, 33)
        self.plotGT = True
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
            plotAbundancesSimple(self.num_endmembers, (self.size[0], self.size[1]), abundances, dict=None,
                                 use_ASC=1,
                                 figure_nr=10)
        return


class PlotafterEpoch(Callback):
    def __init__(self, num_endmembers, size_data, input, endmembersGT, abundances_GT, use_ASC, plot_GT, is_GT_for_A):
        super(PlotafterEpoch, self).__init__()
        self.num_endmembers = num_endmembers
        self.input = input
        self.size_data = size_data
        self.use_ASC = use_ASC
        self.endmembersGT = endmembersGT
        self.abundancesGT = abundances_GT
        self.plotGT = plot_GT
        self.is_GT_for_A = is_GT_for_A
        self.num_epochs = 0
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []
        self.num_epochs = 0

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.num_epochs = epoch
        self.losses.append(logs.get('loss'))
        if epoch % 6 != 0: return
        # return
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer('abundances').output)
        abundances = intermediate_layer_model.predict(self.input)
        endmembers = self.model.layers[len(self.model.layers) - 1].get_weights()[0]
        plotHist(self.losses, 33)
        if self.plotGT:
            dict = order_endmembers(endmembers, self.endmembersGT)
            # if self.is_GT_for_A:
            #     plotAbundances(self.num_endmembers, self.size_data, abundances, self.abundancesGT, dict, self.use_ASC)
            # else:
            #     plotAbundances(self.num_endmembers, self.size_data, abundances, None, None, self.use_ASC, is_GT=False)
            plotEndmembersAndGT(endmembers, self.endmembersGT, dict)
            # plotAbundancesSimple(self.num_endmembers, self.size_data, abundances, dict, use_ASC=1, figure_nr=10)
        else:
            # plotAbundances(self.num_endmembers, self.size_data, abundances, None, None, self.use_ASC)
            plotEndmembers(self.num_endmembers, endmembers)

        print(K.eval(self.model._collected_trainable_weights[-3]))
        return


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# functions

# custom accuracy metric

def normalized_mse_percentage_error(y_true, y_pred):
    y_true = y_true / y_true.max()
    y_pred = y_pred / y_pred.max()
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


# custom loss functions

def SAD(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    sad = -K.mean(y_true2 * y_pred2, axis=-1)

    return sad


def normSAD(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred))
    # sad = -K.log(1.0-K.mean(y_true2 * y_pred2/np.pi, axis=-1))
    sad = K.mean(y_true2 * y_pred2, axis=-1)
    # sid = SID(y_true,y_pred)

    return 0.008 * mse - 1.0 * sad


def normSAD2(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    # sad = -K.log(1.0-K.mean(y_true2 * y_pred2/np.pi, axis=-1))
    sad = K.mean(y_true2 * y_pred2, axis=-1)
    # sid = SID(y_true,y_pred)

    return 0.005 * mse - 0.75 * sad
    # return 0.005*mse-1.0*sad


def numpy_SAD(y_true, y_pred):
    return np.arccos(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))


def SID(y_true, y_pred):
    y_true = K.switch(K.min(y_true) < 0, y_true - K.min(y_true) + K.epsilon(), y_true + K.epsilon())
    y_pred = K.switch(K.min(y_pred) < 0, y_pred - K.min(y_pred) + K.epsilon(), y_pred + K.epsilon())
    p_n = y_true / K.sum(y_true, axis=1, keepdims=True)
    q_n = y_pred / K.sum(y_pred, axis=1, keepdims=True)
    return (K.sum(p_n * K.log(p_n / q_n)) + K.sum(q_n * K.log(q_n / p_n)))


def MSE_KL(y_true, y_pred):
    # y_true=y_true[:,-162:]
    y_true = K.switch(K.min(y_true) < 0, y_true - K.min(y_true) + K.epsilon(), y_true + K.epsilon())
    y_pred = K.switch(K.min(y_pred) < 0, y_pred - K.min(y_pred) + K.epsilon(), y_pred + K.epsilon())
    p_n = y_true / K.max(y_true, axis=1, keepdims=True)
    q_n = y_pred / K.max(y_pred, axis=1, keepdims=True)

    return K.mean(K.square(y_true - y_pred), axis=-1) + 0.5 * (
    K.sum(p_n * K.log(p_n / q_n)) + K.sum((1.001 - p_n) * K.log((1.01 - p_n) / (1.001 - q_n))))


def SCD(y_true, y_pred):
    s_t = y_true - K.mean(y_true, axis=1, keepdims=True)
    s_p = y_pred - K.mean(y_true, axis=1, keepdims=True)

    return 1 - K.mean(K.l2_normalize(s_t + K.epsilon(), axis=-1) * K.l2_normalize(s_p + K.epsilon(), axis=-1))


class DenseTransposeTied(Layer):
    # @interfaces.legacy_dense_support
    def __init__(self, units,
                 tied_to=None,  # Enter a layer as input to enforce weight-tying
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseTransposeTied, self).__init__(**kwargs)
        self.units = units
        # We add these two properties to save the tied weights
        self.tied_to = tied_to
        self.tied_weights = self.tied_to.weights
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        # We remove the weights and bias because we do not want them to be trainable

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        # Return the transpose layer mapping using the explicit weight matrices
        output = K.dot(inputs - self.tied_weights[1], K.transpose(self.tied_weights[0]))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseTransposeTied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def calc_min_SAD(em, encoding_dim):
    SAD_array = np.zeros(shape=(encoding_dim, encoding_dim))
    for i in range(0, encoding_dim):
        for j in range(0, encoding_dim):
            # SAD_array[i,j]=np.arccos(np.sum((em[i,:] * em[j,:]))/np.sqrt(np.sum(em[i,:]^2)*np.sum(em[j,:]^2)))
            SAD_array[i, j] = np.arccos(
                np.sum((em[i, :] * em[j, :])) / np.sqrt(np.sum(em[i, :] * em[i, :]) * np.sum(em[j, :] * em[j, :])))
    return SAD_array


# Computes SID map of reconstructed data
def compute_SID_map(pred, ref):
    SID_map = np.zeros((pred.shape[0], pred.shape[1]))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            SID_map[i, j] = numpy_SID(ref[i, j, :].reshape(-1, 1), pred[i, j, :].reshape(-1, 1))
    return SID_map


def compute_ASAM(endmembers, endmembersGT, dict):
    # Match endmembers to endmembersGT
    normalize = True
    num_endmembers = endmembers.shape[0]
    if dict is None:
        dict = order_endmembers(endmembers, endmembersGT)

    ASAM = 0
    for i in range(num_endmembers):
        # ASAM=ASAM+numpy_SAD(endmembers[i,:],endmembersGT[dict[i],:]) #endmembers[i,:].dot(endmembersGT[dict[i]])/(np.linalg.norm(endmembers[i,:])*np.linalg.norm(endmembersGT[dict[i],:]))
        ASAM = ASAM + np.arccos(endmembers[i, :].dot(endmembersGT[dict[i]]) / (
            np.linalg.norm(endmembers[i, :]) * np.linalg.norm(endmembersGT[dict[i], :])))
    return ASAM / float(num_endmembers)


def compute_ASAM_rad(endmembers, endmembersGT, dict):
    # Match endmembers to endmembersGT
    normalize = True
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
            ASAM = ASAM + compute_SAM_for_endmember(endmembers[i, :], endmembersGT[dict[i]])
            num += 1
    return ASAM / float(num)


def compute_SAM_for_endmember(endmember, endmemberGT):
    return np.arccos(endmember.dot(endmemberGT) / (np.linalg.norm(endmember) * np.linalg.norm(endmemberGT)))
    # return endmember.dot(endmemberGT) / (np.linalg.norm(endmember)*np.linalg.norm(endmemberGT))


def compute_GSME(abundances, abundancesGT, dict):
    num_maps = abundances.shape[2]
    num_pixels = abundances.shape[0] * abundances.shape[1]
    scores = np.zeros(num_maps)
    for i in range(num_maps):
        abundances[:, :, i] = abundances[:, :, i] / abundances[:, :, i].max()
        abundancesGT[:, :, i] = abundancesGT[:, :, i] / abundancesGT[:, :, i].max()
        scores[i] = 1.0 / (num_maps * num_pixels) * np.linalg.norm(
            abundances[:, :, i] - abundancesGT[:, :, dict[i]]) ** 2
    return scores


def extract_patches_from_data(y, batch_size, num_patches):
    patch_dim = int(np.sqrt(batch_size))
    if num_patches > 0:
        patches = extract_patches_2d(y, (patch_dim, patch_dim), num_patches, None)
    else:
        patches = extract_patches_2d(y, (patch_dim, patch_dim))
    return patches


# Function to get the activations from any layer
def get_activations(model, layer_name, input):
    # layer_name = 'abundances'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model.predict(input)


# numpy version of SID to compute SID map
def numpy_SID(y_true, y_pred):
    eps = np.finfo(float).eps
    y_pred = np.where(np.min(y_pred) < 0, y_pred - np.min(y_pred) + eps, y_pred + eps)
    y_true = np.where(np.min(y_true) < 0, y_true - np.min(y_true) + eps, y_true + eps)

    p_n = np.sum(y_true + eps, axis=1, keepdims=True)
    q_n = np.sum(y_pred + eps, axis=1, keepdims=True)

    return np.sum(p_n * np.log(p_n / q_n)) + np.sum(q_n * np.log(q_n / p_n))


def simple_SID(y_true, y_pred):
    y_true = y_true / np.max(y_true)
    y_pred = y_pred / np.max(y_pred)
    p_n = np.sum(y_true)
    q_n = np.sum(y_pred)
    return np.sum(p_n * np.log(p_n / q_n)) + np.sum(q_n * np.log(q_n / p_n))


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


def plot_and_save(encoding_dim, Abundances, endmembers, wvl, output_filename, individual_figs, use_ASC, colormap, hist,
                  Loss):
    if not individual_figs:
        if encoding_dim > 5:
            # Plot endmembers and abundance maps
            n = encoding_dim / 2  # how many digits we will display
            plt.figure(num=1, figsize=(14, 8))

            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                plt.plot(endmembers[i, :], 'r')
                ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                ax = plt.subplot(2, n, i + 1 + n)
                # plt.imshow(S[:, :, i])
                plt.plot(endmembers[i + n, :], 'r')
                # plt.colorbar()
                ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig('endmembers.png')
            # Abundance maps plot
            n = encoding_dim / 2  # how many digits we will display

            plt.figure(num=2, figsize=(14, 8))
            AA = np.sum(Abundances, axis=-1)
            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
                if use_ASC:
                    im = ax.imshow(Abundances[:, :, i] / np.max(Abundances[:, :, i]), cmap=colormap)

                    # plt.imshow(np.greater(Abundances[:, :, i]/np.max(Abundances[:, :, i]),0.7), cmap='jet')
                else:
                    im = ax.imshow(Abundances[:, :, i] / AA, cmap=colormap)
                cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax = plt.subplot(2, n, i + 1 + n)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
                if use_ASC:
                    im = ax.imshow(Abundances[:, :, i + n] / np.max(Abundances[:, :, i + n]), cmap=colormap)
                    # plt.imshow(np.greater(Abundances[:, :, i+n]/np.max(Abundances[:, :, i+n]), 0.7), cmap='jet')
                else:
                    im = ax.imshow(Abundances[:, :, i + n] / AA, cmap=colormap)
                plt.colorbar(im, cax=cax, orientation='horizontal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.savefig('abundances.png')
            plt.figure(3)
            plt.plot(hist.history['loss'])
            plt.show()


        else:
            n = encoding_dim
            plt.figure(num=1, figsize=(14, 8))
            AA = np.sum(Abundances, axis=-1)
            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                plt.plot(endmembers[i, :], 'r')
                ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)
                ax = plt.subplot(2, n, i + n + 1)
                if use_ASC:
                    plt.imshow(Abundances[:, :, i] / np.max(Abundances[:, :, i]), cmap=colormap)
                    # plt.imshow(np.greater(Abundances[:, :, i]/np.max(Abundances[:, :, i]),0.7), cmap='jet')
                else:
                    plt.imshow(Abundances[:, :, i] / AA, cmap=colormap)
                plt.colorbar(orientation='horizontal')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.show()
            plt.savefig('abundances.png')
    else:
        def my_formatter(x, pos):
            if x.is_integer():
                return str(int(x))
            else:
                return str(x)

        formatter = FuncFormatter(my_formatter)
        for i in range(0, encoding_dim):

            plt.figure(num=1, figsize=(12, 7))
            e = np.zeros(shape=(210, 1)).flatten()
            x = wvl[0, :]
            mask = wvl[1, :] == 1
            e[mask] = endmembers[i]
            E = np.ma.masked_where(e == 0, e)
            x = x[4:196]
            e = e[4:196]
            E = E[4:196]
            # plt.plot(x,e,color='lightgray')
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rc('xtick', labelsize=45)
            plt.rc('ytick', labelsize=45)
            plt.plot(x, E, color='blue', linewidth=7.5)
            ymin, ymax = plt.ylim()
            plt.ylim((0, ymax))
            ax = plt.gca()
            ax.yaxis.set_major_formatter(formatter)
            plt.fill_between(x, 0, max(e), where=wvl[1, 4:196] == 0, interpolate=True, facecolor='lightgray', alpha=0.6)
            plt.xlabel('wavelength [nm]', fontsize=48)
            plt.ylabel('intensity', fontsize=48)
            # ax = plt.gca()
            # [i.set_linewidth(0.2) for i in ax.spines.itervalues()]
            if Loss == SID:
                imgname = 'sid_endmember' + '-' + str(i + 1) + '.png'
            if Loss == 'mse':
                imgname = 'mse_endmember' + '-' + str(i + 1) + '.png'
            plt.tight_layout()
            plt.savefig(imgname)
            plt.close()
        for i in range(0, encoding_dim):
            plt.figure(num=1, figsize=(14, 8))
            AA = np.sum(Abundances, axis=-1)
            ax = plt.gca()
            plt.rc('xtick', labelsize=18)
            plt.rc('ytick', labelsize=18)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(position='bottom', size='4%', pad=0.05)

            if use_ASC:
                im = ax.imshow(Abundances[:, :, i] / np.max(Abundances[:, :, i]), cmap=colormap)

                # plt.imshow(np.greater(Abundances[:, :, i]/np.max(Abundances[:, :, i]),0.7), cmap='jet')
            else:
                im = ax.imshow(Abundances[:, :, i] / AA, cmap='jet')

            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.xaxis.set_major_formatter(formatter)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if Loss == SID:
                imgname = 'sid_abundance_map' + '-' + str(i + 1) + '.png'
            if Loss == 'mse':
                imgname = 'mse_abundance_map' + '-' + str(i + 1) + '.png'
            plt.savefig(imgname)
            plt.close()
            # plt.fill_between(x,e,E,where=E>=e,facecolor='lightgray',alpha=0.9)


def plot(num_endmembers, abundances, endmembers, wvl, use_ASC, colormap, Loss, hist):
    def my_formatter(x, pos):
        if x.is_integer():
            return str(int(x))
        else:
            return str(x)

    n = num_endmembers / 2
    plt.figure(num=1, figsize=(14, 8))
    AA = np.sum(abundances, axis=-1)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        if use_ASC:
            im = ax.imshow(abundances[:, :, i] / np.max(abundances[:, :, i]), cmap=colormap)

            # plt.imshow(np.greater(Abundances[:, :, i]/np.max(Abundances[:, :, i]),0.7), cmap='jet')
        else:
            im = ax.imshow(abundances[:, :, i] / AA, cmap=colormap)
        plt.colorbar(im, cax=cax, orientation='horizontal')

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        if use_ASC:
            im = ax.imshow(abundances[:, :, i + n] / np.max(abundances[:, :, i + n]), cmap=colormap)
            # plt.imshow(np.greater(Abundances[:, :, i+n]/np.max(Abundances[:, :, i+n]), 0.7), cmap='jet')
        else:
            im = ax.imshow(abundances[:, :, i + n] / AA, cmap=colormap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()

    formatter = FuncFormatter(my_formatter)
    n = num_endmembers / 2  # how many digits we will display

    plt.figure(num=2, figsize=(14, 8))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, -162:], 'r')
        ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(S[:, :, i])
        plt.plot(endmembers[i + n, -162:], 'r')
        # plt.colorbar()
        ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.figure(3)
    plt.plot(hist.history['loss'])
    plt.show()


def plotAbundances(num_endmembers, size_data, abundances, abundancesGT, dict, use_ASC, figure_nr=10, Title=None,
                   is_GT=False):
    abundances = np.reshape(abundances, (size_data[1], size_data[0], num_endmembers))
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    # if dict is not None and is_GT==False:
    #     # scores = compute_GSME(abundances, abundancesGT, dict)
    # else:
    scores = None
    n = num_endmembers / 2
    if num_endmembers % 2 != 0: n = n + 1
    fig = plt.figure(num=figure_nr, figsize=(14, 8))
    if Title is not None:
        st = plt.suptitle(Title)
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        if is_GT:
            NormedAbundances = abundances[:, :, dict[i]] / AA
        else:
            NormedAbundances = abundances[:, :, i] / AA
        if use_ASC:
            if is_GT:
                im = ax.imshow(abundances[:, :, dict[i]] / np.max(abundances[:, :, dict[i]]), cmap='jet')
            else:
                im = ax.imshow(abundances[:, :, i] / np.max(abundances[:, :, i]), cmap='jet')
                # im = ax.imshow(abundances[:, :, i], cmap='jet')
        else:
            im = ax.imshow(NormedAbundances / np.max(NormedAbundances), cmap='jet')
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if scores is not None:
            ax.set_title("GSME(A): " + format(scores[i], '.4f'))

    plt.tight_layout()
    if Title is not None:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.88)

    plt.draw()
    plt.pause(0.001)


def plotAbundancesSimple(num_endmembers, size_data, abundances, dict, use_ASC=1, figure_nr=16):
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


def plotGTEndmembers(endmembers, figure_nr):
    num_endmembers = endmembers.shape[0]  # how many digits we will display
    n = num_endmembers / 2
    if num_endmembers % 2 != 0: n = n + 1
    plt.figure(num=figure_nr, figsize=(14, 8))
    plt.clf()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, :], 'k')
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def spectrum_length(spectrum):
    apts = [(i, spectrum[i]) for i in range(0, spectrum.shape[0])]
    lengths = np.sqrt(np.sum(np.diff(apts, axis=0) ** 2, axis=1))  # Length between corners
    return np.sum(lengths)


def plotEndmembersAndGT(endmembersGT, endmembers, dict, normalize=True, figure_nr=11):
    num_endmembers = endmembers.shape[0]
    n = num_endmembers / 2  # how many digits we will display
    if num_endmembers % 2 != 0: n = n + 1
    if dict is None:
        dict = order_endmembers(endmembers, endmembersGT)
    aSAM = compute_ASAM_rad(endmembers, endmembersGT, dict)
    fig = plt.figure(num=figure_nr, figsize=(14, 8))
    plt.clf()
    title = "aSAM score for all endmembers: " + format(aSAM, '.3f') + " degrees"
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


def plotHist(loss, fig_nr):
    plt.figure(num=fig_nr, figsize=(14, 8))
    plt.clf()
    plt.plot(loss)
    plt.draw()
    plt.pause(0.001)


def loadGT(filename):
    input = sio.loadmat(filename)
    A = np.asarray(input['A'], dtype=np.float32)
    M = np.asarray(input['M'], dtype=np.float32)
    return {'M': M, 'A': A}


def add_noise(data, SNR, bandwidth):
    L = np.min(data.shape)
    npx = np.max(data.shape)
    sigma = np.sqrt(np.sum(np.sum(np.multiply(data, data))) / npx / L / 10 ** (SNR / 10.0))
    noise = sigma * np.random.randn(data.shape[0], data.shape[1])
    arr = np.arange(L).reshape(L, 1)
    filter_coef = np.exp(-np.multiply(arr, arr) / 2 / bandwidth ** 2)
    filter_coef = filter_coef.T
    # print('add noise filter coef: ',filter_coef.shape)
    scale = np.sqrt(float(L) / np.sum(np.multiply(filter_coef, filter_coef)))
    filter_coef = scale * filter_coef
    noise = idct(np.multiply(dct(noise, norm='ortho'), np.matlib.repmat(filter_coef, npx, 1)), norm='ortho')
    return data + noise


class TV(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            # X=K.eval(x)
            diff = x[1:] - x[:-1]
            regularization += K.mean(K.sqrt(diff ** 2 + 0.000001))

        return regularization * self.l1

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}
