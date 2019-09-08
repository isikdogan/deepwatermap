import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def running_recall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + K.epsilon())
    return recall

def running_precision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (TP_FP + K.epsilon())
    return precision

def running_f1(y_true, y_pred):
    precision = running_precision(y_true, y_pred)
    recall = running_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def focal_loss(y_true, y_pred, alpha=0.25, gamma=1):
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    positive = -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred), axis=-1) * alpha
    negative = -K.sum(K.pow(y_pred, gamma) * (1. - y_true) * K.log(1. - y_pred), axis=-1) * (1-alpha)
    return positive + negative

def adaptive_maxpool_loss(y_true, y_pred, alpha = 0.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    positive = -y_true * K.log(y_pred) * alpha
    negative = -(1. - y_true) * K.log(1. - y_pred) * (1-alpha)
    pointwise_loss = positive + negative
    max_loss = tf.keras.layers.MaxPool2D(pool_size=8, strides=1, padding='same')(pointwise_loss)
    x = pointwise_loss * max_loss
    x = K.mean(x, axis=-1)
    return x

# TODO: try using lovasz loss

def _cumsum(x, axis=None):
    return np.cumsum(x, axis=axis)

def lovasz_grad(gt_sorted):
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.py_func(_cumsum, [gt_sorted], tf.float32)
    union = gts + tf.py_func(_cumsum, [1. - gt_sorted], tf.float32)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_hinge_flat(logits, labels):
    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss

def lovasz_hinge(logits, labels):
    def treat_image(log_lab):
        log, lab = log_lab
        log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
        log, lab = tf.reshape(log, (-1,)), tf.reshape(lab, (-1,))
        return lovasz_hinge_flat(log, lab)
    losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
    loss = tf.reduce_mean(losses)
    return loss

def keras_lovasz_hinge(labels,logits):
    return lovasz_hinge(logits, labels)