import numpy as np
import scipy.io as sio
import h5py as hdf
from skimage.util import view_as_blocks
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans


class HSI:
    def __init__(self, filename):
        self.n_pixels = 0
        self.n_bands = 0
        self.n_rows = 0
        self.n_cols = 0
        self.data = None
        self.test_data = None
        self.size = None
        self.orig_data = None
        self.GT = None
        self.end = None
        self.bands_to_use = None
        self.patch_size = None
        self.S = None
        self.data_path = filename

    def load_data(self, normalize=True, bands_to_skip=None, bands_to_use=None, shuffle=False):
        try:
            data = sio.loadmat(self.data_path)
        except NotImplementedError:
            data = hdf.File(self.data_path, 'r')

        Y = np.asarray(data['Y'], dtype=np.float32)
        if Y.shape[0] < Y.shape[1]:
            Y = Y.transpose()
        self.n_bands = Y.shape[1]
        self.n_rows = data['lines'].item()
        self.n_cols = data['cols'].item()
        self.size = [self.n_cols, self.n_rows]
        self.n_pixels = self.n_cols * self.n_rows

        if 'GT' in data.keys():
            self.GT = data['GT']
            self.is_GT = True
        else:
            self.is_GT = False

        # Preprocess data
        if normalize:
            Y = Y / np.max(Y.flatten())
            self.orig_data = Y  # np.reshape(Y,(self.size[0],self.size[1],self.n_bands))

        if shuffle:
            Y = np.random.permutation(Y)

        self.orig_data = Y
        self.data = Y

        if bands_to_use is not None or bands_to_skip is not None:
            self.bands_to_use = bands_to_use
            if bands_to_skip is not None:
                all_bands = range(Y.shape[1])
                self.bands_to_use = list(set(all_bands) - set(bands_to_skip))
            self.data = self.data[:, :, self.bands_to_use]

        # self.data = np.reshape(Y, (self.size[0], self.size[1], self.n_bands))

    def filter_zeros(self):
        # self.data = self.data[~(self.data < 1e-7).all(axis=2)]
        self.data[self.data < 1e-7] = 0

    def make_patches(self, n, num_patches, use_orig = False):
        if use_orig:
            data = self.orig_data
        else:
            data = self.data
        if n > 1:
            self.patch_size = n
            if self.S is not None:
                stacked = np.concatenate((self.S, data), axis=-1)
                patches = extract_patches_2d(stacked, (n, n), num_patches)
                self.S = patches[:, :, :, 0:self.S.shape[-1]]
                data = patches[:, :, :, -self.n_bands:]
            else:
                data = extract_patches_2d(data, (n, n), num_patches)

            s = data.shape
            data = data.reshape(s[0], n * n, self.n_bands)
        else:
            data = data[np.random.randint(0, self.n_pixels, num_patches)]
        self.data = data

    def make_patches_for_locations(self, n, loc):
        self.patch_size = n
        self.data = np.reshape(self.orig_data, (self.n_cols, self.n_rows, self.n_bands))
        xy = []
        for i in loc:
            line = i // self.n_cols
            col = i % self.n_cols
            if line < (self.n_rows - self.patch_size) and col < (self.n_cols - self.patch_size):
                xy.append((col, line))
        patches = []  # np.zeros((len(xy),n,n,self.n_bands))
        for i in range(0, len(xy)):
            location = xy[i]
            patches.append(self.data[location[0]:location[0] + n, location[1]:location[1] + n, :])
        self.data = np.array(patches)
        s = self.data.shape
        self.data = self.data.reshape(s[0], n * n, self.n_bands)

    def perform_clustering(self, n_classes):
        self.n_classes = n_classes
        clusterer = KMeans(n_clusters=self.n_classes)
        if len(self.orig_data.shape) > 2:
            data = np.reshape(self.orig_data,
                              (self.orig_data.shape[0] * self.orig_data.shape[1], self.orig_data.shape[2]))
        else:
            data = self.orig_data
        clusterer.fit_transform(data)
        self.labels = clusterer.labels_
        unique, counts = np.unique(self.labels, return_counts=True)
        num = np.min(counts)  # np.min([np.min(counts),850//n_classes])
        print(dict(zip(unique, counts)))
        idx = []
        for i in range(self.n_classes):
            labels = np.where(self.labels == i)[0]
            # clust_i = clusterer.transform(self.data.data)[:, i]
            # clust_i = dist[:,i]
            # closest = np.argsort(clust_i)[:num]
            tmp = np.random.choice(labels, num, replace=False)
            idx = np.append(idx, tmp)
        idx = idx.astype(np.int32)
        self.data = self.orig_data[idx]
        self.n_pixels = self.data.shape[0]
        return idx
        # def order_randomly(self,n):
        #     idx=np.random.permutation(range(self.n_pixels))
        #     self.data = self.data[idx]
        #     self.data = self.data.reshape()
