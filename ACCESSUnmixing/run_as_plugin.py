from tqdm import tqdm  # _notebook as tqdm
from pathlib import Path
from configparser import ConfigParser
from warnings import warn
from scipy import io as spio
import os
import numpy as np
import tensorflow as tf
from HySpUn import mse, improvement_only, save_config

from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras import backend as K

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.fmin import generate_trials_to_calculate
from functools import reduce

if __package__ == "ACCESSUnmixing":
    from ACCESSUnmixing.ACCESSUnmixing import Autoencoder
    from ACCESSUnmixing.unmixing.HSI import HSI
    from ACCESSUnmixing.unmixing.losses import SAD
    from ACCESSUnmixing.sveinn.calc_SAD import calc_SAD_2
else:
    from ACCESSUnmixing import Autoencoder
    from unmixing.HSI import HSI
    from unmixing.losses import SAD
    from ACCESSUnmixing.sveinn.calc_SAD import calc_SAD_2

random_seed = 42
tf.set_random_seed(random_seed)
np.random.seed(random_seed)

def run_method(data, resdir, num_runs):
    dataset = data.dataset_name
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__)))
    configpath = os.path.join(__location__, 'datasets.cfg')
    config = ConfigParser()
    config.read(configpath)
    if not config.has_section(dataset):
        warn('No settings found for ' + dataset + ', using defaults.')
        config.add_section(dataset)

    datapath = data.data_path
    init_endmembers = data.init_endmembers
    activation = LeakyReLU(0.2)

    opt_name = config.get(dataset, 'optimizer')
    opt_cfg = dict(config._sections[dataset + ' ' + opt_name])
    for key, value in opt_cfg.items():
        opt_cfg[key] = float(value)
    optimizer = { 'class_name': opt_name, 'config': opt_cfg }

    l2 = config.getfloat(dataset, 'l2')
    l1 = config.getfloat(dataset, 'l1')
    num_patches = config.getint(dataset, 'num_patches')
    epochs = config.getint(dataset, 'epochs')
    batch_size = config.getint(dataset, 'batch_size')
    plot_every_n = config.getint(dataset, 'plot_every_n')
    n_band, n_end = init_endmembers.shape

    my_data = HSI(datapath)
    results = []

    for i in tqdm(range(num_runs), desc="Runs", unit="runs"):
        my_data.load_data(normalize=True, shuffle=False)
        unmixer = Autoencoder(n_end=n_end, data=my_data, activation=activation,
                              optimizer=optimizer, l2=l2, l1=l1, plot_every_n = plot_every_n)
        unmixer.create_model(SAD)
        my_data.make_patches(1, num_patches=num_patches, use_orig=True)
        history = unmixer.fit(epochs=epochs, batch_size=batch_size)
        resfile = 'Run_' + str(i + 1) + '.mat'
        endmembers = unmixer.get_endmembers().transpose()
        abundances = unmixer.get_abundances().reshape(data.n_rows,data.n_cols,data.n_endmembers).transpose((1, 0, 2))
        resdict = {'endmembers': endmembers,
                   'abundances': abundances,
                   'loss': history.history['loss'],
                   'SAD': history.history['SAD']}
        results.append(resdict)
        spio.savemat(resdir / resfile, results[i])
        del unmixer
        K.clear_session()
    return results

#%%

def opt_method(data, resdir, max_evals):

    dataset = data.dataset_name
    __location__ = os.path.realpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__)))
    datapath = data.data_path
    init_endmembers = data.init_endmembers

    n_band, n_end = init_endmembers.shape

    def objective_func(data, hyperpars):
        data.load_data(normalize=True, shuffle=False)

        activation = LeakyReLU(0.2)

        unmixer = Autoencoder(n_end=n_end, data=my_data, activation=activation,
                              optimizer=hyperpars['optimizer'], l2=hyperpars['l2'], l1=hyperpars['l1'], plot_every_n=0)

        unmixer.create_model(SAD)
        my_data.make_patches(1, num_patches=hyperpars['num_patches'], use_orig=True)
        history = unmixer.fit(epochs=100, batch_size=hyperpars['batch_size'])

        endmembers = unmixer.get_endmembers().transpose()
        abundances = unmixer.get_abundances()
        Y = np.transpose(data.orig_data)
        GT = np.transpose(data.GT)
        sad, idx_org, idx_hat, sad_k_m, s0 = calc_SAD_2(GT, endmembers)
        MSE = mse(Y, endmembers, np.transpose(abundances))
        abundances = abundances.reshape(data.n_rows, data.n_cols, endmembers.shape[1]).transpose((1, 0, 2))
        resdict = {'endmembers': endmembers,
                   'abundances': abundances,
                   'loss': history.history['loss'],
                   'SAD': sad,
                   'MSE': MSE}

        del unmixer
        K.clear_session()

        return {'loss': sad, 'status': STATUS_OK, 'attachments': resdict}


    space = {
        'optimizer': {'class_name': 'RMSprop', 'config': {'lr': hp.qloguniform('ACCESS_' + dataset + '_lr', -16, -1, 1e-7)}},
        'l1': hp.qloguniform('ACCESS_' + dataset + '_l1', -16, 2, 1e-7),
        'l2': hp.qloguniform('ACCESS_' + dataset + '_l2', -16, 2, 1e-7),
        'num_patches': scope.int(hp.quniform('ACCESS_' + dataset + '_num_patches', 8, 8192, 1)),
        'batch_size': scope.int(hp.quniform('ACCESS_' + dataset + '_batch_size', 1, 50, 1))
    }

    my_data = HSI(datapath)

    trials = generate_trials_to_calculate([{
        'ACCESS_' + dataset + '_lr': 0.001,
        'ACCESS_' + dataset + '_l1': 0,
        'ACCESS_' + dataset + '_l2': 0,
        'ACCESS_' + dataset + '_num_patches': 1028,
        'ACCESS_' + dataset + '_batch_size': 32
    }])

    pars = fmin(lambda x: objective_func(my_data, x),
                space=space,
                trials=trials,
                algo=tpe.suggest,
                max_evals=max_evals,
                rstate=np.random.RandomState(random_seed))

    improvements = reduce(improvement_only, trials.losses(), [])

    save_config(resdir, dataset, pars, trials.average_best_error())

    return improvements, pars, trials
