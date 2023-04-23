import run_as_plugin
import numpy as np
from pathlib import Path
from sveinn.HSID import HSID

init_endmembers = np.load('../HyMeCoT/initials/Samson/endmembers.npy')
init_abundances = np.load('../HyMeCoT/initials/Samson/abundances.npy')
hsid = HSID('../../Datasets/Samson.mat',init_endmembers=init_endmembers,
            init_abundances=init_abundances, dataset_name='Samson')
run_as_plugin.keyra(hsid,Path('./Results/'),1)
