# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:52:09 2020

@author: JIn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:39:21 2020

@author: JIn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:25:16 2019

@author: jin
"""
import os
import sys
import scipy.io as sio
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras.constraints import non_neg
from tensorflow.python.keras.layers import LeakyReLU, Input, Dense, BatchNormalization, GaussianDropout
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers as optimizers
from tqdm import tqdm
from keras_tqdm import TQDMCallback
from tensorflow import set_random_seed
from scipy.io import loadmat, savemat
from tensorflow.python.keras.layers.core import Lambda
from absl import flags
from absl import app
from datetime import datetime
from tensorflow.python.ops import control_flow_ops
# pdist--> pairwise distance between two points in n-dim space
# square form ---> convert a vector-form distance vector to a square-form distance matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import normalize
from numpy import linalg as LA
from tensorflow.python.keras.callbacks import Callback
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
if __package__ == "ACCESSUnmixing":
    from ACCESSUnmixing.unmixing.HSI import HSI
    from ACCESSUnmixing.unmixing.losses import SAD
    from ACCESSUnmixing.unmixing.plotting import PlotWhileTraining
    from ACCESSUnmixing.frosti import utils
else:
    from unmixing.HSI import HSI
    from unmixing.losses import SAD,SID,normMSE,normSAD,SIDSAD
    from unmixing.plotting import PlotWhileTraining
    from frosti import utils
np.set_printoptions(threshold=np.inf)


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("adversarial", False, "Use Adversarial Autoencoder or regular Autoencoder")
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("reconstruct", False, "Reconstruct image")
flags.DEFINE_bool("generate", False, "Generate image from latent")
flags.DEFINE_bool("generate_grid", False, "Generate grid of images from latent space (only for 2D latent)")
flags.DEFINE_bool("plot", False, "Plot latent space")
flags.DEFINE_integer("latent_dim", 5, "Latent dimension")

# Train
flags.DEFINE_integer("epochs", 500, "Number of training epochs")
flags.DEFINE_integer("train_samples", 3600, "Number of training samples from MNIST")
flags.DEFINE_integer("batchsize",40 , "Training batchsize")

# Test
flags.DEFINE_integer("test_samples", 3600, "Number of test samples from MNIST")
flags.DEFINE_list("latent_vec", None, "Latent vector (use with --generate flag)")




#x=mat_contents['Y_T']
fname = 'Synthetic_Access_SNR10.mat'
mat_contents=loadmat('Synthetic_Access_SNR10.mat')
x=mat_contents['Y']
x_c=mat_contents['E_bundles']
E=mat_contents['E_VCA']

l_vca=0.5
l_1=0
use_bias = False
activation_set=LeakyReLU(0.2)
initializer = initializers.glorot_normal()
optimizer=optimizers.Adam(0.0001)
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
set_random_seed(random_seed)
rand_x = np.random.RandomState(42)
rand_y = np.random.RandomState(42)



def E_reg(weight_matrix):
        return l_vca*SAD(weight_matrix,E)+l_1*tf.reduce_mean(tf.matmul(tf.transpose(weight_matrix,perm=[1,0]),weight_matrix))

def custom_loss_wrapper(abundances):
    def custom_loss(y_true, y_pred):
        return SAD(y_true,y_pred)+l_1*tf.norm(abundances,ord=1)#+l_lap*A_lp
    return custom_loss



def create_model(input_dim, latent_dim, verbose=True, save_graph=False):
    inputs = Input(shape=(input_dim,))
    ###########encoder###########################################################
    encoder = Sequential()
    encoder.add(Dense(intermediate_dim1, input_shape=(input_dim,), activation='relu',name='Dense_1'))
    encoder.add(Dense(latent_dim * 9, input_shape=(input_dim,),use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=activation_set))
            # en coded = BatchNormalization()(encoded)
    encoder.add(Dense(latent_dim * 6, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=activation_set))
    encoder.add(Dense(latent_dim*3, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=activation_set))
    encoder.add(Dense(latent_dim, use_bias=use_bias, kernel_regularizer=None, kernel_initializer=None,
                            activation=activation_set))
    encoder.add(BatchNormalization())
        # Soft Thresholding
    encoder.add(utils.SparseReLU(alpha_initializer='zero', alpha_constraint=non_neg(), activity_regularizer=None))
        # Sum To One (ASC)
    encoder.add(utils.SumToOne(axis=0, name='abundances', activity_regularizer=None))

    #########################################Decoder############################
    decoder = Sequential()

    decoder.add(Dense(input_dim, input_shape=(latent_dim,), activation='linear', name='endmembers', use_bias=use_bias,
                     kernel_constraint=non_neg(), kernel_regularizer=E_reg, kernel_initializer=initializer))
    encoder_output = encoder(inputs)
    autoencoder = Model(autoencoder_input, decoder(encoder_output))
    autoencoder.compile(optimizer, loss=SAD)
    return autoencoder, encoder, decoder
    autoencoder.compile(optimizer,loss=SAD)
    if verbose:
        print("Autoencoder Architecture")
        print(autoencoder.summary())



    return autoencoder,encoder,decoder

    
def train(n_samples, batch_size, n_epochs):
        autoencoder_base, encoder_base, decoder_base = create_model(input_dim=input_dim, latent_dim=FLAGS.latent_dim)
        inputs_a = Input(shape=(input_dim))
        inputs_b = Input(shape=(input_dim))
        encoder_output_a = encoder_base(inputs_a)
        encoder_output_b = encoder_base(inputs_b)

        decoder_output_a = decoder_base(encoder_output_a)
        decoder_output_b = decoder_base(encoder_output_b)
        autoencoder = Model([inputs_a, inputs_b], [decoder_output_a, decoder_output_b])
        autoencoder.compile(optimizer,loss=SAD)

        autoencoder.fit(x=[x, x_c], y=[x, x_c],epochs=n_epochs, batch_size=batch_size, verbose=0,callbacks=[evaluator],shuffle=True)

        E=decoder_base.get_layer('endmembers').get_weights()[0]
        A,_= encoder_base.predict(x, x_c)
        Y,_=autoencoder.predict(x, x_c)
        return A,Y,E



def main(argv):
    global desc,intermediate_dim1,intermediate_dim2,intermediate_dim3
    original_dim=200
    intermediate_dim1 = int(np.ceil(original_dim*1.2) + 5)
    intermediate_dim2 = int(max(np.ceil(original_dim/4), FLAGS.latent_dim+2) + 3)
    intermediate_dim3 = int(max(np.ceil(original_dim/10), FLAGS.latent_dim+1))
    if FLAGS.adversarial:
        desc = "aae"
    else:
        desc = "regular"
    if FLAGS.train:
        A,Y,E=train(n_samples=FLAGS.train_samples, batch_size=FLAGS.batchsize, n_epochs=FLAGS.epochs)
        savemat('results_synthetic_bundles'+'batch'+str(FLAGS.batchsize)+'.mat', {'A':A,'Y':Y,'E':E})





if __name__ == "__main__":
    app.run(main)
    
