# -*- coding: utf-8 -*-
"""This tensorflow extension implements an MTRNN.
for tensorflow 1.4 keras
Heinrich et al. 2020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.recurrent \
    import _generate_dropout_mask, _generate_zero_filled_state_for_cell, RNN
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, gen_array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


_MAX_TIMESCALE = 999999
_MAX_SIGMA = 500000
_ALMOST_ONE = 0.999999
_ALMOST_ZERO = 0.000001
_DEBUGMODE = False

@tf_export('keras.layers.MTRNNCell')
class AMTRNNCell(Layer):
    """Cell class for Adaptive Multiple Timescale Recurrent Neural Network.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
            important: csc size and values are steered via
            the initial_states (or final states)
        h_layers: Positive integer, number of horizontal layers.
            The dimensionality of the outputspace is a concatenation of
            all h_layers k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: connection scheme in case of more than one h_layers
            Default: `adjacent`
            Other options are `clocked`, and `dense`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_kernel: Boolean, whether the layer has weighted input;
            for the MTRNN the default is that information are fed via
            the recurrent weihts only
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 h_layers=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_kernel=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(AMTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.h_layers = len(units_vec)
            self.IO_units = units_vec[0]
            self.all_units = 0
            for k in range(self.h_layers):
                self.all_units += units_vec[k]
        else:
            self.all_units = units_vec
            if h_layers is not None and h_layers > 1:
                self.h_layers = int(h_layers)
                self.units_vec = [units_vec//h_layers
                                  for k in range(self.h_layers)]
                self.IO_units = self.units_vec[0]
            else:
                self.h_layers = 1
                self.units_vec = [units_vec]
                self.IO_units = units_vec
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.h_layers:
                raise ValueError("vector of tau must be of same size as "
                                 "h_layers or size of vector of num_units")
            for k in  range(self.h_layers):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., tau_vec[k])] for k in range(self.h_layers)
                 for n in range(self.units_vec[k])],
                dtype=array_ops.dtypes.float32, shape=[self.all_units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.h_layers > 1:
                self.tau_vec = [tau_vec for k in range(self.h_layers)]
            else:
                self.tau_vec = [tau_vec]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=array_ops.dtypes.float32,
                shape=[self.all_units], name="taus")

        if self.IO_units < self.all_units:
            self.is_subset_IO_units = True
            self.non_IO_units = self.all_units-self.IO_units
        else:
            self.is_subset_IO_units = False
            self.non_IO_units = 0

        self.activation = activations.get(activation)
        self.use_kernel = use_kernel
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_tau_initializer = initializers.get(w_tau_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.w_tau_regularizer = regularizers.get(w_tau_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_tau_constraint = constraints.get(w_tau_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.all_units, self.all_units)
        self.output_size = self.IO_units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        #input_dim = input_shape[-1]

        if self.use_kernel:
            self.kernel = self.add_weight(
                shape=(self.IO_units, self.IO_units),
                name='kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)
        else:
            self.kernel = None

        if self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.h_layers):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.h_layers]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'dense':
            self.recurrent_kernel_vec = [self.add_weight(
                shape=(self.all_units, self.all_units),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint)]
        else:  # == 'adjacent'
            self.recurrent_kernel_vec = []
            for k in range(self.h_layers):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.h_layers, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.all_units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.w_tau = self.add_weight(
            shape=(self.all_units,),
            name='wtimescales',
            initializer=self.w_tau_initializer,
            regularizer=self.w_tau_regularizer,
            constraint=self.w_tau_constraint)

        self.log_taus = K.log(self.taus - array_ops.constant(_ALMOST_ONE))
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs
        # the initial (or final) states of the csc units
        # are given (or obtained) from the prev_y (or y)
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        if self.kernel is not None:
            if self.is_subset_IO_units:
                h = array_ops.concat(
                    [K.dot(x, self.kernel),
                     array_ops.zeros(
                         (array_ops.shape(x)[0], self.non_IO_units),
                         dtype=x.dtype)], -1)
            else:
                h = K.dot(x, self.kernel)
        else:
            if self.is_subset_IO_units:
                h = array_ops.concat(
                    [x,
                     array_ops.zeros(
                         (array_ops.shape(x)[0], self.non_IO_units),
                         dtype=x.dtype)], -1)
            else:
                h = x

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.h_layers], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.h_layers)], 1)
        elif self.connectivity == 'dense':
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])
        else:  # == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.h_layers, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.h_layers)], 1)

        taus_act = K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        if _DEBUGMODE:
            x = gen_array_ops.check_numerics(x, 'AMTRNNCell: Numeric error for x')
            prev_y = gen_array_ops.check_numerics(prev_y, 'AMTRNNCell: Numeric error for prev_y')
            prev_z = gen_array_ops.check_numerics(prev_z, 'AMTRNNCell: Numeric error for prev_z')
            h = gen_array_ops.check_numerics(h, 'AMTRNNCell: Numeric error for h')
            r = gen_array_ops.check_numerics(r, 'AMTRNNCell: Numeric error for r')
            y = gen_array_ops.check_numerics(y, 'AMTRNNCell: Numeric error for y')
            z = gen_array_ops.check_numerics(z, 'AMTRNNCell: Numeric error for z')

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True

        # return array_ops.slice(
        #     y, [0, 0], [array_ops.shape(x)[0], self.IO_units]), [y, z]

        """
        We modified this in a way that the z is sliced within the 
        AMTRNNcb and not directly in the cell.
        This way it is possible to also read out the activations.
        """
        return y, [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE if self.built else None

    def get_config(self):
        config = {
            'IO_units':
                self.IO_units,
            'units_vec':
                self.units_vec,
            'h_layers':
                self.h_layers,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_kernel':
                self.use_kernel,
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AMTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.MTRNNCell')
class AMTRNNcaCell(Layer):
    """Cell class for Adaptive Multiple Timescale Recurrent Neural Network
       with context bias.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
            important: csc size and values are steered via
            the initial_states (or final states)
        csc_units: Positive integer,
            number of context controlling units.
        h_layers: Positive integer, number of horizontal layers.
            The dimensionality of the outputspace is a concatenation of
            all h_layers k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: connection scheme in case of more than one h_layers
            Default: `adjacent`
            Other options are `clocked`, and `dense`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_kernel: Boolean, whether the layer has weighted input;
            for the MTRNN the default is that information are fed via
            the recurrent weihts only
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 csc_units,
                 h_layers=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_kernel=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(AMTRNNcaCell, self).__init__(**kwargs)
        self.csc_units = csc_units
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.h_layers = len(units_vec)
            self.IO_units = units_vec[0]
            self.all_units = 0
            for k in range(self.h_layers):
                self.all_units += units_vec[k]
        else:
            self.all_units = units_vec
            if h_layers is not None and h_layers > 1:
                self.h_layers = int(h_layers)
                self.units_vec = [units_vec//h_layers
                                  for k in range(self.h_layers)]
                self.IO_units = self.units_vec[0]
            else:
                self.h_layers = 1
                self.units_vec = [units_vec]
                self.IO_units = units_vec
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.h_layers:
                raise ValueError("vector of tau must be of same size as "
                                 "h_layers or size of vector of num_units")
            for k in  range(self.h_layers):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., tau_vec[k])] for k in range(self.h_layers)
                 for n in range(self.units_vec[k])],
                dtype=array_ops.dtypes.float32, shape=[self.all_units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.h_layers > 1:
                self.tau_vec = [tau_vec for k in range(self.h_layers)]
            else:
                self.tau_vec = [tau_vec]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=array_ops.dtypes.float32,
                shape=[self.all_units], name="taus")

        if self.IO_units < self.all_units:
            self.is_subset_IO_units = True
            self.non_IO_units = self.all_units-self.IO_units
        else:
            self.is_subset_IO_units = False
            self.non_IO_units = 0

        self.activation = activations.get(activation)
        self.use_kernel = use_kernel
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_tau_initializer = initializers.get(w_tau_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.w_tau_regularizer = regularizers.get(w_tau_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_tau_constraint = constraints.get(w_tau_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.all_units, self.all_units)
        self.output_size = self.IO_units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        #input_dim = input_shape[-1]

        if self.use_kernel:
            self.kernel = self.add_weight(
                shape=(self.IO_units, self.IO_units),
                name='kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)
        else:
            self.kernel = None

        if self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.h_layers):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.h_layers]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'dense':
            self.recurrent_kernel_vec = [self.add_weight(
                shape=(self.all_units, self.all_units),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint)]
        else:  # == 'adjacent'
            self.recurrent_kernel_vec = []
            for k in range(self.h_layers):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.h_layers, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.all_units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.w_tau = self.add_weight(
            shape=(self.all_units,),
            name='wtimescales',
            initializer=self.w_tau_initializer,
            regularizer=self.w_tau_regularizer,
            constraint=self.w_tau_constraint)

        self.log_taus = K.log(self.taus - _ALMOST_ONE)
         # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs
        # the initial (or final) states of the csc units
        # are given (or obtained) from the prev_y (or y)
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        if self.kernel is not None:
            if self.is_subset_IO_units:
                h = array_ops.concat(
                    [K.dot(x, self.kernel),
                     array_ops.zeros(
                         (array_ops.shape(x)[0], self.non_IO_units),
                         dtype=x.dtype)], -1)
            else:
                h = K.dot(x, self.kernel)
        else:
            if self.is_subset_IO_units:
                h = array_ops.concat(
                    [x,
                     array_ops.zeros(
                         (array_ops.shape(x)[0], self.non_IO_units),
                         dtype=x.dtype)], -1)
            else:
                h = x

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.h_layers], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.h_layers)], 1)
        elif self.connectivity == 'dense':
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])
        else:  # == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.h_layers, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.h_layers)], 1)

        taus_act = K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        if _DEBUGMODE:
            x = gen_array_ops.check_numerics(x, 'AMTRNNcaCell: Numeric error for x')
            prev_y = gen_array_ops.check_numerics(prev_y, 'AMTRNNcaCell: Numeric error for prev_y')
            prev_z = gen_array_ops.check_numerics(prev_z, 'AMTRNNcaCell: Numeric error for prev_z')
            h = gen_array_ops.check_numerics(h, 'AMTRNNcaCell: Numeric error for h')
            r = gen_array_ops.check_numerics(r, 'AMTRNNcaCell: Numeric error for r')
            y = gen_array_ops.check_numerics(y, 'AMTRNNcaCell: Numeric error for y')
            z = gen_array_ops.check_numerics(z, 'AMTRNNcaCell: Numeric error for z')

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True

        """
        We modified this in a way that the z is sliced within the 
        AMTRNNca and not directly in the cell.
        This way it is possible to also read out the activations.
        """
        return z, [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE if self.built else None

    def get_config(self):
        config = {
            'IO_units':
                self.IO_units,
            'csc_units':
                self.csc_units,
            'units_vec':
                self.units_vec,
            'h_layers':
                self.h_layers,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_kernel':
                self.use_kernel,
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AMTRNNcaCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.AMTRNN')
class AMTRNN(RNN):
    """Adaptive Multiple Timescale Recurrent Neural Network that can have
       multiple horizontal layers, where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
            important: csc size and values are steered via
            the initial_states (or final states)
        h_layers: Positive integer, number of horizontal layers.
            The dimensionality of the outputspace is a concatenation of
            all h_layers k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: connection scheme in case of more than one h_layers
            Default: `adjacent`
            Other options are `clocked`, `partitioned`, and `dense`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_kernel: Boolean, whether the layer has weighted input;
            for the MTRNN the default is that information are fed via
            the recurrent weihts only
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 h_layers=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_kernel=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `MultipleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = AMTRNNCell(
            units_vec,
            h_layers=h_layers,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_kernel=use_kernel,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_tau_initializer=w_tau_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_tau_regularizer=w_tau_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_tau_constraint=w_tau_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(AMTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(AMTRNN, self).call(inputs, mask=mask,
                                        training=training,
                                        initial_state=initial_state)

    @property
    def IO_units(self):
        return self.cell.IO_units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def h_layers(self):
        return self.cell.h_layers

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_kernel(self):
        return self.cell.use_kernel

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def w_tau_initializer(self):
        return self.cell.w_tau_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def w_tau_regularizer(self):
        return self.cell.w_tau_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def w_tau_constraint(self):
        return self.cell.w_taus_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'IO_units':
                self.IO_units,
            'units_vec':
                self.units_vec,
            'h_layers':
                self.h_layers,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_kernel':
                self.use_kernel,
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AMTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


@tf_export('keras.layers.AMTRNNca')
class AMTRNNca(RNN):
    """Adaptive Multiple Timescale Recurrent Neural Network
       with Context Abstraction, that can have multiple horizontal layers,
       where an abstraction is obtained from an input sequence.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
            important: csc size and values are steered via
            the initial_states (or final states)
        csc_units: Positive integer,
            number of context controlling units.
        h_layers: Positive integer, number of horizontal layers.
            The dimensionality of the outputspace is a concatenation of
            all h_layers k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: connection scheme in case of more than one h_layers
            Default: `adjacent`
            Other options are `clocked`, `partitioned`, and `dense`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_kernel: Boolean, whether the layer has weighted input;
            for the AMTRNN the default is that information are fed via
            the recurrent weihts only
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 csc_units,
                 h_layers=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_kernel=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=True,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `MultipleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = AMTRNNcaCell(
            units_vec,
            csc_units,
            h_layers=h_layers,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_kernel=use_kernel,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_tau_initializer=w_tau_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_tau_regularizer=w_tau_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_tau_constraint=w_tau_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(AMTRNNca, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        z_out = super(AMTRNNca, self).call(inputs, mask=mask,
                                            training=training,
                                            initial_state=initial_state)

        result = array_ops.slice(
                 z_out[:,-1], [0, self.cell.all_units - self.cell.csc_units],
                 [array_ops.shape(z_out[:,-1])[0], self.cell.csc_units])

        if _DEBUGMODE:
            result = gen_array_ops.check_numerics(result, 'AMTRNNca: Numeric error for result')

        """
        StH 31.10.2019:
        The returned z_out, if return_sequences is True, 
        returns the output of all MTRNN neurons for all time steps.
        For the output of the MTRNNca, we only consider the last time step
        of the csc units.
        Nevertheless, we return both at this point, 
        to be able to read out the hidden activations later.
        Thus for using it as an MTRNNca only provide output[0] to the next layer!
        """
        return [result, z_out]

    def get_output(self):
        return self.output[0]

    def get_activations(self):
        return self.output[1]

    def get_taus(self):
        return self.cell.get_taus()

    @property
    def IO_units(self):
        return self.cell.IO_units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def h_layers(self):
        return self.cell.h_layers

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_kernel(self):
        return self.cell.use_kernel

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def w_tau_initializer(self):
        return self.cell.w_tau_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def w_tau_regularizer(self):
        return self.cell.w_tau_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def w_tau_constraint(self):
        return self.cell.w_taus_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'IO_units':
                self.IO_units,
            'csc_units':
                self.csc_units,
            'units_vec':
                self.units_vec,
            'h_layers':
                self.h_layers,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_kernel':
                self.use_kernel,
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AMTRNNca, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


@tf_export('keras.layers.AMTRNNcb')
class AMTRNNcb(RNN):
    """Adaptive Multiple Timescale Recurrent Neural Network that can have
       multiple horizontal layers, where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
            important: csc size and values are steered via
            the initial_states (or final states)
        seqs_max_length: Positive integer,
            maximal length of the generated sequences.
        h_layers: Positive integer, number of horizontal layers.
            The dimensionality of the outputspace is a concatenation of
            all h_layers k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau: Positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: connection scheme in case of more than one h_layers
            Default: `adjacent`
            Other options are `clocked`, `partitioned`, and `dense`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_kernel: Boolean, whether the layer has weighted input;
            for the MTRNN the default is that information are fed via
            the recurrent weihts only
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 seqs_max_length,
                 h_layers=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_kernel=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=True,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `MultipleCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = AMTRNNCell(
            units_vec,
            h_layers=h_layers,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_kernel=use_kernel,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_tau_initializer=w_tau_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_tau_regularizer=w_tau_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_tau_constraint=w_tau_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(AMTRNNcb, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.sequences_max_length = seqs_max_length
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, ini_csc_vals, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None

        inputs = array_ops.zeros(
            (array_ops.shape(ini_csc_vals)[0],
             self.sequences_max_length, self.cell.IO_units),
            dtype=ini_csc_vals.dtype)
        if initial_state is not None:
            [y, z] = initial_state  #states
        else:
            [y, z] = self.cell.get_initial_state(
                batch_size=array_ops.shape(ini_csc_vals)[0],
                dtype=ini_csc_vals.dtype)

        # the initial csc values are the input to the network, but are fed
        # only the some of the units (csc units) and only at timestep 0.
        z = array_ops.concat(
            [array_ops.slice(z, [0, 0], [
                -1, self.cell.all_units - array_ops.shape(ini_csc_vals)[1]]),
                ini_csc_vals], 1)

        z_out = super(AMTRNNcb, self).call(
            inputs, mask=mask, training=training,
            initial_state=[y, z])

        result = array_ops.slice(
            z_out, [0, 0, 0], [array_ops.shape(inputs)[0], array_ops.shape(inputs)[1], self.cell.IO_units])

        if _DEBUGMODE:
            result = gen_array_ops.check_numerics(result, 'AMTRNNcb: Numeric error for result')

        """
        The returned z_out, returns the output of all MTRNN neurons for all time steps.
        For the output of the MTRNNcb, 
        we only consider the IO units but over the whole sequence.
        Nevertheless, we return both at this point, 
        to be able to read out the hidden activations later.
        Thus for using it as an MTRNNcb only provide output[0] to the next layer!
        """
        return [result, z_out]
        #return result[0] #only return the output but not the states

    def get_output(self):
        return self.output[0]

    def get_activations(self):
        return self.output[1]

    def get_taus(self):
        return self.cell.get_taus()

    @property
    def IO_units(self):
        return self.cell.IO_units

    @property
    def seqs_max_length(self):
        return self.sequences_max_length

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def h_layers(self):
        return self.cell.h_layers

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_kernel(self):
        return self.cell.use_kernel

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def w_tau_initializer(self):
        return self.cell.w_tau_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def w_tau_regularizer(self):
        return self.cell.w_tau_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def w_tau_constraint(self):
        return self.cell.w_taus_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        #TODO: check whether we need to take vals from the cell instead
        config = {
            'IO_units':
                self.IO_units,
            'seqs_max_length':
                self.sequences_max_length,
            'units_vec':
                self.units_vec,
            'h_layers':
                self.h_layers,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_kernel':
                self.use_kernel,
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(AMTRNNcb, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)
