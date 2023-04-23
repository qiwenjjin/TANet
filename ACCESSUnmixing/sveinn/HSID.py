import numpy as np
import scipy.io as sio
import h5py as hdf
from typing import Any, List
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class HSID:
    data_path: str
    dataset_name: str = 'unknown'
    data_var_name: str = 'Y'
    size: tuple = ()
    n_bands: int = None
    n_rows: int = None
    n_cols: int = None
    n_pixels: int = None
    n_endmembers: int = None
    bands_last: bool = False
    freq_list: List[float] = field(default_factory=list)
    data: np.ndarray = None
    orig_data: np.ndarray = None
    has_reference: bool = False
    ref_path: str = None
    ref_var_names: tuple = ('GT', 'S_GT')
    ref_endmembers: np.ndarray = None
    ref_abundances: np.ndarray = None
    init_endmembers: np.ndarray = None
    init_abundances: np.ndarray = None
    bands_to_use: List[int] = field(default_factory=list)
    S: Any = None

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        if self.ref_path is not None:
            self.ref_path = Path(self.ref_path)
        if self.data is None:
            self.load_data()

    def load_data(self, normalize=False, bands_to_skip=None):
        try:
            data = sio.loadmat(self.data_path)
        except NotImplementedError:
            data = hdf.File(self.data_path, 'r')

        y = np.asarray(data[self.data_var_name], dtype=np.float32)
        self.orig_data = y

        if y.shape[0] > y.shape[1]:
            y = y.transpose()
        self.n_bands = y.shape[0]
        if 'lines' in data:
            self.n_rows = data['lines'].item()
            self.n_cols = data['cols'].item()
            self.size = (self.n_cols, self.n_rows)
            self.n_pixels = self.n_cols * self.n_rows

        if self.ref_path:
            ref_data = sio.loadmat(self.ref_path)
            if self.ref_var_names[0] in ref_data:
                self.ref_endmembers = data[self.ref_var_names[0]]
                self.has_reference = True
                if self.ref_endmembers.shape[0] < self.ref_endmembers.shape[1]:
                    self.ref_endmembers = self.ref_endmembers.transpose()
                if self.ref_var_names[1] in data:
                    self.ref_abundances = data[self.ref_var_names[1]]
                self.n_endmembers = self.ref_endmembers.shape[1]
        elif self.ref_var_names[0] in data:
            self.ref_endmembers = data[self.ref_var_names[0]]
            if self.ref_endmembers.shape[0] < self.ref_endmembers.shape[1]:
                self.ref_endmembers = self.ref_endmembers.transpose()
            self.has_reference = True
            if self.ref_var_names[1] in data:
                self.ref_abundances = data[self.ref_var_names[1]]

        if self.init_endmembers.shape[0] < self.init_endmembers.shape[0]:
            self.init_endmembers = self.init_endmembers.transpose()

        # Preprocess data
        if normalize:
            y = y / np.max(y.flatten())

        if self.bands_to_use or bands_to_skip:
            if bands_to_skip is not None:
                all_bands = range(y.shape[1])
                self.bands_to_use = list(set(all_bands) - set(bands_to_skip))
            self.data = self.data[:, :, self.bands_to_use]

        if self.bands_last:
            y = y.transpose()
            self.ref_endmembers = self.ref_endmembers.transpose()
            self.init_endmembers = self.init_endmembers.transpose()

        self.data = y

    def filter_zeros(self):
        self.data[self.data < 1e-7] = 0
