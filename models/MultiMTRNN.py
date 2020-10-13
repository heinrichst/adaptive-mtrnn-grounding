# -*- coding: utf-8 -*-
"""This tensorflow extension implements an Multi-MTRNN architecture.
for tensorflow 1.4 keras
Heinrich et al. 2020
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import models.keras_extend.mtrnn as keras_layers_mtrnn
import models.keras_extend.xmtrnn as keras_layers_xmtrnn


def MultiMTRNN(mtrnn_type='adaptive',
                    ut_size=[44, 80, 23],
                    ut_tau=[2, 5, 50],
                    ut_csc_size=12,
                    au_size=[13, 40, 23],
                    au_tau=[2, 20, 700],
                    au_csc_size=12,
                    sm_size=[16, 40, 23],
                    sm_tau=[2, 20, 700],
                    sm_csc_size=12,
                    vi_size=[19, 40, 23],
                    vi_tau=[2, 20, 700],
                    vi_csc_size=12,
                    output_softmax=False,
                    rec_initializer='orthogonal',
                    mtrnn_connectivity='adjacent',
                    au_length_max=100,
                    sm_length_max=100,
                    vi_length_max=100,
                    ut_length_max=100
               ):
    """
    Multi-modal Multiple Timescales Recurrent Neural Network
    :param mtrnn_type:
    :param ut_size:
    :param ut_tau:
    :param ut_csc_size:
    :param au_size:
    :param au_tau:
    :param au_csc_size:
    :param sm_size:
    :param sm_tau:
    :param sm_csc_size:
    :param vi_size:
    :param vi_tau:
    :param vi_csc_size:
    :param output_softmax:
    :param rec_initializer:
    :param mtrnn_connectivity:
    :param au_length_max:
    :param sm_length_max:
    :param vi_length_max:
    :param ut_length_max:
    :return: the tensorflow-keras model
    """
    inp_k = 0
    inp_s = []
    csc_s = []
    if au_size is not None:
        inp_k += 1
        au_input = keras.layers.Input(shape=(au_length_max, au_size[0]),
                                      name="input_"+str(inp_k))
        inp_s += [au_input]

        au_mtrnn_name = mtrnn_name("au_mtrnna_", au_size, mtrnn_connectivity)
        if (mtrnn_type == "adaptive"):
            au_mtrnn = keras_layers_xmtrnn.AMTRNNca(
                au_size, au_csc_size,
                tau_vec=au_tau,
                use_kernel=False, recurrent_initializer=rec_initializer,
                connectivity=mtrnn_connectivity, name=au_mtrnn_name)
            au_csc = au_mtrnn(au_input)[0]
        else:  #== "conventional"
            au_mtrnn = keras_layers_mtrnn.MTRNNca(
                au_size, au_csc_size,
                tau_vec=au_tau,
                use_kernel=False, recurrent_initializer=rec_initializer,
                connectivity=mtrnn_connectivity, name=au_mtrnn_name)
            au_csc = au_mtrnn(au_input)
        csc_s += [au_csc]

    if sm_size is not None:
        inp_k += 1
        sm_input = keras.layers.Input(shape=(sm_length_max, sm_size[0]),
                                      name="input_"+str(inp_k))
        inp_s += [sm_input]

        sm_mtrnn_name = mtrnn_name("sm_mtrnna_", sm_size, mtrnn_connectivity)
        if (mtrnn_type == "adaptive"):
            sm_mtrnn = keras_layers_xmtrnn.AMTRNNca(
                sm_size, sm_csc_size,
                tau_vec=sm_tau,
                use_kernel=False, recurrent_initializer=rec_initializer,
                connectivity=mtrnn_connectivity, name=sm_mtrnn_name)
            sm_csc = sm_mtrnn(sm_input)[0]
        else:  #== "conventional"
            sm_mtrnn = keras_layers_mtrnn.MTRNNca(
                sm_size, sm_csc_size,
                tau_vec=sm_tau,
                use_kernel=False, recurrent_initializer=rec_initializer,
                connectivity=mtrnn_connectivity, name=sm_mtrnn_name)
            sm_csc = sm_mtrnn(sm_input)
        csc_s += [sm_csc]

    if vi_size is not None:
        inp_k += 1
        vi_input = keras.layers.Input(shape=(vi_length_max, vi_size[0]),
                                      name="input_"+str(inp_k))
        inp_s += [vi_input]

        vi_mtrnn_name = mtrnn_name("vi_mtrnna_", vi_size, mtrnn_connectivity)
        if (mtrnn_type == "adaptive"):
            vi_mtrnn = keras_layers_xmtrnn.AMTRNNca(
                vi_size, vi_csc_size,
                tau_vec=vi_tau,
                use_kernel=False, recurrent_initializer=rec_initializer,
                connectivity=mtrnn_connectivity, name=vi_mtrnn_name)
            vi_csc = vi_mtrnn(vi_input)[0]
        else:  #== "conventional"
            vi_mtrnn = keras_layers_mtrnn.MTRNNca(
                vi_size, vi_csc_size,
                tau_vec=vi_tau,
                use_kernel=False, recurrent_initializer=rec_initializer,
                connectivity=mtrnn_connectivity, name=vi_mtrnn_name)
            vi_csc = vi_mtrnn(vi_input)
        csc_s += [vi_csc]

    vi_sm_ut_name = "cell_assembly"
    if len(csc_s) > 1:
        ca_concat = keras.layers.concatenate(csc_s)
    else:
        ca_concat = csc_s[0]
    ut_csc = keras.layers.Dense(ut_csc_size, name=vi_sm_ut_name)(ca_concat)

    ut_mtrnn_name = mtrnn_name("ut_mtrnnb_", ut_size, mtrnn_connectivity)
    if (mtrnn_type == "adaptive"):
        ut_mtrnn = keras_layers_xmtrnn.AMTRNNcb(
            ut_size, ut_length_max,
            tau_vec=ut_tau,
            use_kernel=False, recurrent_initializer=rec_initializer,
            connectivity=mtrnn_connectivity, name=ut_mtrnn_name)
        ut_IO = ut_mtrnn(ut_csc)[0]
    else:  #== "conventional"
        ut_mtrnn = keras_layers_mtrnn.MTRNNcb(
            ut_size, ut_length_max,
            tau_vec=ut_tau,
            use_kernel=False, recurrent_initializer=rec_initializer,
            connectivity=mtrnn_connectivity, name=ut_mtrnn_name)
        ut_IO = ut_mtrnn(ut_csc)

    ut_out_name = "ut_output"
    if output_softmax:
        ut_output = keras.layers.Softmax(name=ut_out_name)(ut_IO)
    else:
        ut_output = ut_IO

    model = keras.Model(inp_s, ut_output)

    return model


def mtrnn_name(name, size, connectivity):
    for k in range(len(size)):
        name += str(size[k]) + "_"
    name += connectivity[0:5]
    return name


def MM_intermediate_in_csc(model, modality, size, mtrnn_connectivity):
    full_mtrnn_name = mtrnn_name("" + modality + "_mtrnna_", size, mtrnn_connectivity)
    in_csc_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(full_mtrnn_name).output[0])
    return in_csc_model


def MM_intermediate_ut_csc(model):
    vi_sm_ut_name = "cell_assembly"
    ut_csc_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(vi_sm_ut_name).output)
    return ut_csc_model


def MM_intermediate_in_act(model, modality, size, mtrnn_connectivity):
    full_mtrnn_name = mtrnn_name("" + modality + "_mtrnna_", size, mtrnn_connectivity)
    in_act_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(full_mtrnn_name).output[1])
    return in_act_model


def MM_intermediate_ut_act(model, ut_size, mtrnn_connectivity):
    full_mtrnn_name = mtrnn_name("ut_mtrnnb_", ut_size, mtrnn_connectivity)
    ut_act_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(full_mtrnn_name).output[1])
    return ut_act_model


def MM_intermediate_out_debug(model, ut_size, mtrnn_connectivity):
    ut_mtrnn_name = mtrnn_name("ut_mtrnnb_", ut_size, mtrnn_connectivity)
    ut_out_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(ut_mtrnn_name).output[0])
    return ut_out_model


# Main, needed only for debug:
if __name__ == '__main__':

    model = MultiMTRNN(ut_size=[3,4,2],
                        ut_tau=[2,5,70],
                        ut_csc_size=2,
                        sm_size=[3,4,2],
                        sm_tau=[2,5,50],
                        sm_csc_size=2,
                        vi_size=[2,5,50],
                        vi_tau=[2,5,70],
                        vi_csc_size=2,
                        mtrnn_connectivity='adjacent',
                        mtrnn_type='adaptive'
                       )

    model.summary()
    keras.utils.plot_model(model)
