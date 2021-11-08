import tensorflow.contrib.keras.api.keras.backend as K
import tensorflow.contrib.keras.api.keras.backend as KC
import numpy as np
import tensorflow as T


def SAD(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    # sad = T.acos(K.mean(y_true2 * y_pred2, axis=-1))
    sad = T.acos((y_true2 * y_pred2))
    # sad = -K.mean(y_true2 * y_pred2, axis=-1)
    return sad


def SIDSAD(y_true, y_pred):
    sad = SAD(y_true, y_pred)
    sid = SID(y_true, y_pred)
    return sid * T.sin(sad)


def normSAD(y_true, y_pred):
    # y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    # y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred))
    # sad = -K.log(1.0-K.mean(y_true2 * y_pred2/np.pi, axis=-1))
    sad = SAD(y_true, y_pred)
    # sid = SID(y_true,y_pred)

    # return 0.008*mse-1.0*sad
    return 0.008 * mse + 1.0 * sad


def normMSE(y_true, y_pred):
    y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred))
    return mse


def normSAD2(y_true, y_pred):
    # y_true2 = K.l2_normalize(y_true + K.epsilon(), axis=-1)
    # y_pred2 = K.l2_normalize(y_pred + K.epsilon(), axis=-1)
    mse = K.mean(K.square(y_true - y_pred))
    sad = SAD(y_true, y_pred)
    # sad = -K.log(1.0-SAD(y_true, y_pred)/np.pi)
    # sid = SID(y_true,y_pred)

    # return 0.005 * mse + 0.75 * sad
    return 0.005 * mse + 10.0 * sad


def SID(y_true, y_pred):
    y_true = K.switch(K.min(y_true) < 0, y_true - K.min(y_true) + K.epsilon(), y_true + K.epsilon())
    y_pred = K.switch(K.min(y_pred) < 0, y_pred - K.min(y_pred) + K.epsilon(), y_pred + K.epsilon())
    p_n = y_true / K.sum(y_true, axis=1, keepdims=True)
    q_n = y_pred / K.sum(y_pred, axis=1, keepdims=True)
    return (K.sum(p_n * K.log(p_n / q_n)) + K.sum(q_n * K.log(q_n / p_n)))


class DSSIMObjective():
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
               not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = KC.backend()

    def __int_shape(self, x):
        return KC.int_shape(x) if self.backend == 'tensorflow' else KC.shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a gradient definition in the Theano tree
        #   and cannot be used for learning

        kernel = [self.kernel_size, self.kernel_size]
        y_true = KC.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = KC.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))

        patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid', self.dim_ordering)
        patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid', self.dim_ordering)

        # Reshape to get the var in the cells
        bs, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = KC.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = KC.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = KC.mean(patches_true, axis=-1)
        u_pred = KC.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)


def SID(y_true, y_pred):
    y_true = K.switch(K.min(y_true) < 0, y_true - K.min(y_true) + K.epsilon(), y_true + K.epsilon())
    y_pred = K.switch(K.min(y_pred) < 0, y_pred - K.min(y_pred) + K.epsilon(), y_pred + K.epsilon())
    p_n = y_true / K.sum(y_true, axis=-1, keepdims=True)
    q_n = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
    return (K.sum(p_n * K.log(p_n / q_n)) + K.sum(q_n * K.log(q_n / p_n)))
