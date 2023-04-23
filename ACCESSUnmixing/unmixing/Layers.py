from tensorflow.contrib.keras.api.keras.layers import Layer
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
from tensorflow.contrib.keras.api.keras import initializers, regularizers, activations, constraints, optimizers
import tensorflow as tf
from tensorflow.python.keras.layers import Layer, InputSpec, ThresholdedReLU

from tensorflow.python.keras.utils import get_custom_objects


def _moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)
    elif K.backend() == 'theano':
        import theano.tensor as T

        mean_batch = T.mean(x, axis=axes, keepdims=keep_dims)
        var_batch = T.var(x, axis=axes, keepdims=keep_dims)
        return mean_batch, var_batch
    else:
        raise RuntimeError("Currently does not support CNTK backend")


class CustomRegularization(Layer):
    def __init__(self, l1, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)
        self.l1 = l1

    def call(self, x):
        loss = tf.reduce_sum(tf.abs(x), axis=-1)
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.norm(loss, axis=-1)
        self.add_loss(self.l1 * loss, x)
        # you can output whatever you need, just update output_shape adequately
        # But this is probably useful
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class SparseReLU(Layer):
    def __init__(self, alpha_initializer='zeros',
                 activity_regularizer=None,
                 alpha_constraint=None,
                 **kwargs):
        super(SparseReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_constraint = constraints.get(alpha_constraint)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        self.alpha = self.add_weight(shape=(1, 1), name='alpha',
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

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint)
        }
        base_config = super(SparseLeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SumToOne(Layer):
    def __init__(self, output_dim, scale, lambda1, tv, n, **kwargs):
        self.output_dim = output_dim
        self.scale = scale
        self.lambda1 = lambda1
        self.tv = tv
        self.n = n
        super(SumToOne, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SumToOne, self).build(input_shape)

    def l_one_min_two(self, x, lambda1, n):
        patch_size = n * n
        x = tf.abs(x + K.epsilon())
        l1 = tf.reduce_sum(tf.abs(tf.norm(x, 1, axis=-1) - tf.norm(x, 2, axis=-1)), axis=None)

        regularization = lambda1 * l1
        return 1 / patch_size * regularization

    def hans(self, x, lambda1):
        # x = tf.nn.l2_normalize(x, axis=[1, 2])
        C = []
        for i in range(6):
            B = x[i, :, :, :]
            for j in range(4):
                x = tf.tile(tf.expand_dims(B[:, :, j], 2), [1, 1, 4])
                C.append(K.sum(tf.multiply(x, B) / tf.to_float(tf.size(B))))

        return K.sum(C) * lambda1

    def soft_ASC(self, x, lambda1, lambda2, n):
        patch_size = n * n
        z = x
        z = tf.abs(z + K.epsilon())
        l1 = tf.reduce_sum(tf.norm(z, 0.5, axis=-1), axis=None)
        x = tf.reduce_sum(x, axis=-1)
        ones = tf.ones_like(x)
        # flattened = tf.layers.flatten(x-ones)
        regularization = tf.norm(x - ones, 2)

        return 1 / patch_size * lambda1 * regularization + 1 / patch_size * l1 * lambda2

    def l_half(self, x, lambda1, tv, n):
        patch_size = n * n
        z = tf.abs(x + K.epsilon())
        l1 = tf.reduce_sum(tf.norm(z, 0.5, axis=-1), axis=None)
        # l1=self.l_one_min_two(x,lambda1,x)

        regularization = lambda1 * l1
        return 1 / patch_size * regularization

    def call(self, x):
        # x *= K.cast(x >= K.epsilon(), K.floatx())
        # x = K.abs(x)
        # x = K.relu(x)
        # x = K.transpose(x)
        # x = x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon())
        # x = K.transpose(x)
        x = K.softmax(self.scale * x)
        # a, v = tf.nn.top_k(x, k=4)
        # a = K.min(x, axis=-1, keepdims=True)
        # z = tf.cast(tf.greater_equal(x, a), K.floatx())
        # x = tf.multiply(x, z)
        self.add_loss(self.l_half(x, self.lambda1, self.tv, self.n), x)
        # self.add_loss(self.hans(x,self.lambda1),x)
        # self.add_loss(self.soft_ASC(x,self.lambda1,self.tv,self.n),x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(SumToOne, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

        # Custom nonnegative unit norm constraint


# class Top_K_SumToOne(Layer):
#     def __init__(self, axis=0, activity_regularizer=None, **kwargs):
#         self.axis = axis
#         self.uses_learning_phase = True
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         super(Top_K_SumToOne, self).__init__(**kwargs)
#
#     def call(self, x, mask=None):
#         a,v = top_k(x,k=2)
#         a = K.min(a,axis=1,keepdims=True)
#         b = tf.greater_equal(x,a)
#         x = tf.multiply(x,tf.cast(b,K.floatx()))
#         x *= K.cast(x >= K.epsilon(), K.floatx())
#         x = K.transpose(x)
#         x_normalized = x / (K.sum(x, axis=0) + K.epsilon())
#         x = K.transpose(x_normalized)
#         # x = x*2.0/(K.max(x,axis=-1, keepdims=True))#4.0*x/(K.max(K.flatten(x))+K.epsilon())
#         # return K.softmax(x)
#         return x
#
#     def get_config(self):
#         config = {'axis': self.axis}
#         base_config = super(Top_K_SumToOne, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#         # Custom nonnegative unit norm constraint
#
# def top_2(x,kk):
#     a, v = top_k(x,k=kk)
#     a = K.min(a, axis=1, keepdims=True)
#     b = tf.greater_equal(x, a)
#     x = tf.multiply(x, tf.cast(b, K.floatx()))
#     return x
#
# class Top_K(Layer):
#     def __init__(self, axis=0, k=2, activity_regularizer=None, **kwargs):
#         self.axis = axis
#         self.uses_learning_phase = True
#         self.k = k
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         super(Top_K, self).__init__(**kwargs)
#
#     def call(self, x, mask=None):
#         a,v = top_k(x,k=self.k)
#         a = K.min(a,axis=1,keepdims=True)
#         b = tf.greater_equal(x,a)
#         x = tf.multiply(x,tf.cast(b,K.floatx()))
#         return x
#
#     def get_config(self):
#         config = {'axis': self.axis}
#         base_config = super(Top_K, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#         # Custom nonnegative unit norm constraint

class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'InstanceNormalization': InstanceNormalization})


def _moments(x, axes, shift=None, keep_dims=False):
    ''' Wrapper over tensorflow backend call '''
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.nn.moments(x, axes, shift=shift, keep_dims=keep_dims)
    elif K.backend() == 'theano':
        import theano.tensor as T

        mean_batch = T.mean(x, axis=axes, keepdims=keep_dims)
        var_batch = T.var(x, axis=axes, keepdims=keep_dims)
        return mean_batch, var_batch
    else:
        raise RuntimeError("Currently does not support CNTK backend")


class BatchRenormalization(Layer):
    """Batch renormalization layer (Sergey Ioffe, 2017).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        epsilon: small float > 0. Fuzz parameter.
            Theano expects epsilon >= 1e-5.
        mode: integer, 0, 1 or 2.
            - 0: feature-wise normalization.
                Each feature map in the input will
                be normalized separately. The axis on which
                to normalize is specified by the `axis` argument.
                Note that if the input is a 4D image tensor
                using Theano conventions (samples, channels, rows, cols)
                then you should set `axis` to `1` to normalize along
                the channels axis.
                During training and testing we use running averages
                computed during the training phase to normalize the data
            - 1: sample-wise normalization. This mode assumes a 2D input.
            - 2: feature-wise normalization, like mode 0, but
                using per-batch statistics to normalize the data during both
                testing and training.
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        r_max_value: Upper limit of the value of r_max.
        d_max_value: Upper limit of the value of d_max.
        t_delta: At each iteration, increment the value of t by t_delta.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
            Note that the order of this list is [gamma, beta, mean, std]
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the gamma vector.
        beta_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the beta vector.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/abs/1702.03275)
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self, epsilon=1e-3, mode=0, axis=-1, momentum=0.99,
                 r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_init='zero',
                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None,
                 **kwargs):
        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.r_max_value = r_max_value
        self.d_max_value = d_max_value
        self.t_delta = t_delta
        if self.mode == 0:
            self.uses_learning_phase = True
        super(BatchRenormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name))
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name))
        self.running_mean = self.add_weight(shape=shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        # Note: running_std actually holds the running variance, not the running std.
        self.running_std = self.add_weight(shape=shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        self.r_max = K.variable(np.ones((1,)), name='{}_r_max'.format(self.name))

        self.d_max = K.variable(np.zeros((1,)), name='{}_d_max'.format(self.name))

        self.t = K.variable(np.zeros((1,)), name='{}_t'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        if self.mode == 0 or self.mode == 2:
            assert self.built, 'Layer must be built before being called'
            input_shape = K.int_shape(x)

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            mean_batch, var_batch = _moments(x, reduction_axes, shift=None, keep_dims=False)
            std_batch = (K.sqrt(var_batch + self.epsilon))

            r_max_value = K.get_value(self.r_max)
            r = std_batch / (K.sqrt(self.running_std + self.epsilon))
            r = K.stop_gradient(K.clip(r, 1 / r_max_value, r_max_value))

            d_max_value = K.get_value(self.d_max)
            d = (mean_batch - self.running_mean) / K.sqrt(self.running_std + self.epsilon)
            d = K.stop_gradient(K.clip(d, -d_max_value, d_max_value))

            if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
                x_normed_batch = (x - mean_batch) / std_batch
                x_normed = (x_normed_batch * r + d) * self.gamma + self.beta
            else:
                # need broadcasting
                broadcast_mean = K.reshape(mean_batch, broadcast_shape)
                broadcast_std = K.reshape(std_batch, broadcast_shape)
                broadcast_r = K.reshape(r, broadcast_shape)
                broadcast_d = K.reshape(d, broadcast_shape)
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)

                x_normed_batch = (x - broadcast_mean) / broadcast_std
                x_normed = (x_normed_batch * broadcast_r + broadcast_d) * broadcast_gamma + broadcast_beta

            # explicit update to moving mean and standard deviation
            self.add_update([K.moving_average_update(self.running_mean, mean_batch, self.momentum),
                             K.moving_average_update(self.running_std, std_batch ** 2, self.momentum)], x)

            # update r_max and d_max
            r_val = self.r_max_value / (1 + (self.r_max_value - 1) * K.exp(-self.t))
            d_val = self.d_max_value / (1 + ((self.d_max_value / 1e-3) - 1) * K.exp(-(2 * self.t)))

            self.add_update([K.update(self.r_max, r_val),
                             K.update(self.d_max, d_val),
                             K.update_add(self.t, K.variable(np.array([self.t_delta])))], x)

            if self.mode == 0:
                if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
                    x_normed_running = K.batch_normalization(
                        x, self.running_mean, self.running_std,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)
                else:
                    # need broadcasting
                    broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                    x_normed_running = K.batch_normalization(
                        x, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                # pick the normalized form of x corresponding to the training phase
                # for batch renormalization, inference time remains same as batchnorm
                x_normed = K.in_train_phase(x_normed, x_normed_running)

        elif self.mode == 1:
            # sample-wise normalization
            m = K.mean(x, axis=self.axis, keepdims=True)
            std = K.sqrt(K.var(x, axis=self.axis, keepdims=True) + self.epsilon)
            x_normed_batch = (x - m) / (std + self.epsilon)

            r_max_value = K.get_value(self.r_max)
            r = std / (self.running_std + self.epsilon)
            r = K.stop_gradient(K.clip(r, 1 / r_max_value, r_max_value))

            d_max_value = K.get_value(self.d_max)
            d = (m - self.running_mean) / (self.running_std + self.epsilon)
            d = K.stop_gradient(K.clip(d, -d_max_value, d_max_value))

            x_normed = ((x_normed_batch * r) + d) * self.gamma + self.beta

            # update r_max and d_max
            t_val = K.get_value(self.t)
            r_val = self.r_max_value / (1 + (self.r_max_value - 1) * np.exp(-t_val))
            d_val = self.d_max_value / (1 + ((self.d_max_value / 1e-3) - 1) * np.exp(-(2 * t_val)))
            t_val += float(self.t_delta)

            self.add_update([K.update(self.r_max, r_val),
                             K.update(self.d_max, d_val),
                             K.update(self.t, t_val)], x)

        return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'mode': self.mode,
                  'axis': self.axis,
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                  'momentum': self.momentum,
                  'r_max_value': self.r_max_value,
                  'd_max_value': self.d_max_value,
                  't_delta': self.t_delta}
        base_config = super(BatchRenormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'BatchRenormalization': BatchRenormalization})


class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. Group Normalization's computation is independent
     of batch sizes, and its accuracy is stable in a wide range of batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical to
    Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is equal
    to number of channels), then this operation becomes identical to Instance Normalization.
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                                                                       'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                                                                       'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        mean, variance = _moments(inputs, group_reduction_axes[2:], keep_dims=True)
        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)

        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})
