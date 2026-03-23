# model.py
# Model architecture and loss functions for 2.5D EfficientNet U-Net

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Try EfficientNet B2
try:
    from tensorflow.keras.applications import EfficientNetB2 as TF_EfficientNetB2
except Exception:
    import efficientnet.tfkeras as efn
    TF_EfficientNetB2 = efn.EfficientNetB2

ALPHA_WD = 1e-5
K_INIT = "he_normal"
EPS = 1e-6

# clDice loss and metric (pure-TF)
def _max_pool2d(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 1, 1, 1], padding='SAME')

def soft_skeletonize(x, iterations=20, kernel_size=3, dim=2):
    x = tf.cast(x, tf.float32)
    x = tf.clip_by_value(x, 0.0, 1.0)
    sk = tf.zeros_like(x)
    y = x
    for _ in range(iterations):
        eroded = -_max_pool2d(-y, kernel_size)
        opened = _max_pool2d(eroded, kernel_size)
        delta = tf.nn.relu(y - opened)
        sk = sk + delta
        y = eroded
    sk = tf.clip_by_value(sk, 0.0, 1.0)
    return sk

def cldice_score(y_true, y_pred, iterations=20, kernel_size=3, dim=2, eps=1e-6):
    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.cast(y_pred, tf.float32)
    y_t = tf.clip_by_value(y_t, 0.0, 1.0)
    y_p = tf.clip_by_value(y_p, 0.0, 1.0)
    Sg = soft_skeletonize(y_t, iterations=iterations, kernel_size=kernel_size, dim=dim)
    Sp = soft_skeletonize(y_p, iterations=iterations, kernel_size=kernel_size, dim=dim)
    spatial_axes = list(range(1, len(y_t.shape)))
    inter1 = tf.reduce_sum(Sg * y_p, axis=spatial_axes)
    denom1 = tf.reduce_sum(y_p, axis=spatial_axes)
    inter2 = tf.reduce_sum(Sp * y_t, axis=spatial_axes)
    denom2 = tf.reduce_sum(y_t, axis=spatial_axes)
    Tprec = inter1 / (denom1 + eps)
    Tsens = inter2 / (denom2 + eps)
    both_empty = tf.logical_and(tf.equal(denom1, 0.0), tf.equal(denom2, 0.0))
    both_empty_f = tf.cast(both_empty, tf.float32)
    Tprec = Tprec * (1.0 - both_empty_f) + both_empty_f * 1.0
    Tsens = Tsens * (1.0 - both_empty_f) + both_empty_f * 1.0
    cldice = (2.0 * Tprec * Tsens) / (Tprec + Tsens + eps)
    return tf.reduce_mean(cldice)

def cldice_loss(y_true, y_pred, iterations=20, kernel_size=3, dim=2, eps=1e-6):
    return 1.0 - cldice_score(y_true, y_pred, iterations=iterations, kernel_size=kernel_size, dim=dim, eps=eps)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2.0 * inter + smooth) / (denom + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred)

def combined_bce_dice_cldice(y_true, y_pred, cldice_weight=1.0, use_cldice=True, cldice_iterations=20, cldice_kernel=3, dim=2):
    b = bce_loss(y_true, y_pred)
    d = dice_loss(y_true, y_pred)
    cl = cldice_loss(y_true, y_pred, iterations=cldice_iterations, kernel_size=cldice_kernel, dim=dim) if use_cldice else 0.0
    return b + d + cldice_weight * cl

def make_final_loss(cldice_weight=1.0, use_cldice=True, cldice_iterations=20, cldice_kernel=3, dim=2):
    def loss_fn(y_true, y_pred):
        return combined_bce_dice_cldice(y_true, y_pred, cldice_weight=cldice_weight, use_cldice=use_cldice, cldice_iterations=cldice_iterations, cldice_kernel=cldice_kernel, dim=dim)
    return loss_fn

def ConvBnAct(filters, kernel_size=(3,3), strides=(1,1), activation='relu', use_bn=True, name=None):
    def f(x):
        x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer=K_INIT, kernel_regularizer=regularizers.l2(ALPHA_WD))(x)
        if use_bn:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x
    return f

def residual_block(filters):
    def f(x):
        shortcut = x
        in_c = K.int_shape(x)[-1]
        if in_c != filters:
            shortcut = Conv2D(filters, (1,1), padding='same', kernel_initializer=K_INIT, kernel_regularizer=regularizers.l2(ALPHA_WD))(shortcut)
        y = ConvBnAct(filters, (3,3), activation='relu')(x)
        y = ConvBnAct(filters, (3,3), activation=None)(y)
        y = Add()([shortcut, y])
        y = Activation('relu')(y)
        return y
    return f

def deep_supervision_block(inp, upscale_factor=1, name=None):
    x = Conv2D(1, (1,1), padding='same', kernel_initializer=K_INIT, kernel_regularizer=regularizers.l2(ALPHA_WD))(inp)
    x = Activation('sigmoid')(x)
    if upscale_factor != 1:
        x = UpSampling2D(size=(upscale_factor, upscale_factor), interpolation='bilinear')(x)
    return x

def get_2p5d_model(input_height, input_width, n_ch=3, deep_supervision=True, pretrained=True):
    filters = (128, 64, 32, 16, 8)
    backbone = TF_EfficientNetB2(weights='imagenet' if pretrained else None, include_top=False, input_shape=(input_height, input_width, 3))
    skip_names = ("block6a_expand_activation", "block4a_expand_activation", "block3a_expand_activation", "block2a_expand_activation")
    skips = []
    for name in skip_names:
        skips.append(backbone.get_layer(name).output)
    x = backbone.output
    inp0 = backbone.input
    inp_conv = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer=K_INIT, kernel_regularizer=regularizers.l2(ALPHA_WD))(inp0)
    dsvs = []
    for i in range(len(filters)):
        name = f"decoderBlock{i}"
        skip = skips[i] if i < len(skips) else inp_conv
        x = UpSampling2D((2,2), interpolation='bilinear', name=name + "_up")(x)
        x = Concatenate(name=name + "_concat")([x, skip])
        x = ConvBnAct(filters[i], (3,3), activation='relu')(x)
        x = residual_block(filters[i])(x)
        if deep_supervision and i < len(skips):
            dsvs.append(x)
    x = Conv2D(1, (3,3), padding='same', kernel_initializer=K_INIT, kernel_regularizer=regularizers.l2(ALPHA_WD))(x)
    main_out = Activation('sigmoid', name='main_out')(x)
    interim_model = Model(inputs=backbone.input, outputs=[main_out] + dsvs)
    inputs = Input(shape=(input_height, input_width, n_ch), name='input_stack')
    if n_ch == 1:
        inp = Lambda(lambda z: K.tile(z, (1,1,1,3)))(inputs)
    else:
        inp = inputs
    interim_outputs = interim_model(inp)
    outputs = []
    upscale_factors = [16, 8, 4, 2]
    for idx, o in enumerate(interim_outputs):
        if idx == 0:
            outputs.append(Lambda(lambda z: z, name='main_out')(o))
        else:
            upscale = upscale_factors[idx - 1] if (idx - 1) < len(upscale_factors) else 1
            d = deep_supervision_block(o, upscale_factor=upscale, name=f'dsv{idx}')
            outputs.append(d)
    model = Model(inputs=inputs, outputs=outputs)
    return model
