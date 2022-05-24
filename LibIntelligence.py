import tensorflow as tf
import pickle
# import numpy as np
import SimpleITK as sitk

import sklearn
import radiomics


FLOAT_tf = tf.float32
xavier_init = tf.keras.initializers.glorot_normal
constant_init = tf.constant_initializer
var_scale_init = tf.keras.initializers.VarianceScaling
orthogonal_init = tf.keras.initializers.Orthogonal


def save_model(model, tag_model):
    tag_model = tag_model + '.pkl'
    with open(tag_model, 'wb') as file:
        pickle.dump(model, file)


def load_model(tag_model):
    tag_model = tag_model + '.pkl'
    with open(tag_model, 'rb') as file:
        model = pickle.load(file)
    return model


def initialize_radiomics():
    texture_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
        verbose=False
    )
    texture_extractor.disableAllFeatures()
    _text_feat = {
        ckey: [] for ckey in texture_extractor.featureClassNames
    }
    texture_extractor.enableFeaturesByName(**_text_feat)

    print('Extraction parameters:\n\t', texture_extractor.settings)
    print('Enabled filters:\n\t', texture_extractor.enabledImagetypes)
    print('Enabled features:\n\t', texture_extractor.enabledFeatures)


def radiomics_extractor(image, mask):
    texture_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
        verbose=False,
        force2D=False
    )
    texture_extractor.disableAllFeatures()
    _text_feat = {ckey: [] for ckey in texture_extractor.featureClassNames}
    texture_extractor.enableFeaturesByName(**_text_feat)

    return texture_extractor.execute(
        sitk.GetImageFromArray(image),
        sitk.GetImageFromArray(mask)
    )


def evaluate_model(y_test, y_pred):
    print(
        'Model accuracy score : {0:0.9f}'.format(
            sklearn.metrics.accuracy_score(y_test, y_pred)
        )
    )
    print(
        'Model f1 score : {0:0.9f}'.format(
            sklearn.metrics.f1_score(
                y_test,
                y_pred,
                pos_label='positive',
                average='weighted'
            )
        )
    )
    print(
        'Model precision score : {0:0.9f}'.format(
            sklearn.metrics.precision_score(
                y_test,
                y_pred,
                pos_label='positive',
                average='weighted'
            )
        )
    )
    print(
        'Model recall score : {0:0.9f}'.format(
            sklearn.metrics.recall_score(
                y_test,
                y_pred,
                pos_label='positive',
                average='weighted'
            )
        )
    )
    # print(
    #     'Model ROC AUC score : {0:0.9f}'.format(
    #         sklearn.metrics.roc_auc_score(
    #             y,
    #             CV.predict_proba(X),
    #             multi_class='ovr'
    #         )
    #     )
    # )
    # scores = cross_val_score(
    #     CV,
    #     X,
    #     y,
    #     cv=3,
    #     scoring='accuracy'
    # )
    # print(
    #     'CV %0.2f accuracy with a standard deviation of %0.2f' % \
    #         (scores.mean(), scores.std())
    # )


def dense(
  tensor23,
  n_hid,
  activ,
  drop=0.,
  normalizer=None,
  training=True,
  W_init=None,
  use_bias=True,
  scope='dense',
  fix=False
):
    trainable = not fix
    with tf.variable_scope(scope):
        input_shape = tensor23.shape.as_list()
        batch_size = input_shape[0]
        input_dim = input_shape[-1]
        if len(input_shape) == 3:
            tensor23 = tf.reshape(tensor23, [-1, input_dim])

        if W_init is None:
            W_init = var_scale_init(factor=0.01, mode='FAN_AVG', uniform=False)
            # W_init = orthogonal_init(gain=0.01, dtype=FLOAT_tf)
            W = tf.get_variable(
                'W',
                [input_dim, n_hid],
                dtype=FLOAT_tf,
                initializer=W_init,
                trainable=trainable
            )

        logits = tf.matmul(tensor23, W)
        if normalizer is None and use_bias:
            b_init = constant_init(0.)
            b = tf.get_variable(
                'b', [n_hid],
                dtype=FLOAT_tf,
                initializer=b_init,
                trainable=trainable
            )
            logits = tf.add(logits, b)
            output = (logits if activ is None else activ(logits))
            output = tf.nn.dropout(output, keep_prob=1.-drop)
        if len(input_shape) == 3:
            output = tf.reshape(output, [batch_size, -1, n_hid])
            print(scope, output.shape)
        return output


def conv2d(
    tensor4D, kernel2D, stride2D, out_dim, activ, drop=0.,
    normalizer=None, training=True, padding='SAME',
    W_init=None, use_bias=True, scope='conv2d', fix=False
):
    trainable = not fix
    with tf.variable_scope(scope):
        xs = tensor4D.shape.as_list()
        if len(xs) == 3:
            tensor4D = tf.expand_dims(tensor4D, axis=-1)
            input_dim = 1
        elif len(xs) == 4:
            input_dim = xs[-1]  # tensor34.shape.as_list()[-1]
        else:
            raise ValueError('Input dims must be 3 or 4 !!')

        stride = [1] + stride2D + [1]

        if W_init is None:
            # W_init = xavier_init(uniform=False)
            W_init = var_scale_init(factor=0.01, mode='FAN_AVG', uniform=False)
            # W_init = orthogonal_init(gain=0.01, dtype=FLOAT_tf)
            W = tf.get_variable(
                'W',
                kernel2D + [input_dim, out_dim],
                dtype=FLOAT_tf,
                initializer=W_init,
                trainable=trainable
            )

        logits = tf.nn.conv2d(tensor4D, W, stride, padding)
        if normalizer is None and use_bias:
            b_init = constant_init(0.)
            b = tf.get_variable(
                'b',
                [out_dim],
                dtype=FLOAT_tf,
                initializer=b_init,
                trainable=trainable
            )
            logits = tf.nn.bias_add(logits, b, name='logits')
        if normalizer == 'BN':
            logits = batch_norm(logits, training=training, trainable=trainable)
            # logits = tf.compat.v1.estimator.layers.batch_norm(
            #     logits,
            #     center=True,
            #     scale=True,
            #     is_training=training
            # )
            # logits = tf.layers.batch_normalization(
            #     logits,
            #     axis=-1,
            #     momentum=0.99,
            #     epsilon=0.001,
            #     center=True,
            #     scale=True,
            #     training=training,
            #     trainable=trainable
            # )
            output = (logits if activ is None else activ(logits))
            output = tf.nn.dropout(output, keep_prob=1.-drop)
            # print(scope, output.shape)
        return output


def pool2d(ptype, tensor34, kernel2D, stride2D=None, pad='SAME', sqz=False):
    # ptype: 'MAX', 'AVG'
    xs = tensor34.shape.as_list()
    if len(xs) == 3:
        tensor34 = tf.expand_dims(tensor34, axis=-1)

    if stride2D is None:
        stride2D = kernel2D

    if ptype == 'MAX':
        output = tf.layers.max_pooling2d(
            tensor34,
            kernel2D,
            stride2D,
            pad,
            data_format='channels_last'
        )
        # output = tf.nn.max_pool(
        #     tensor34,
        #     [1]+kernel2D+[1],
        #     [1]+stride2D+[1],
        #     pad
        # )
    if ptype == 'AVG':
        output = tf.layers.average_pooling2d(
            tensor34,
            kernel2D,
            stride2D,
            pad,
            data_format='channels_last'
        )
        # output = tf.nn.avg_pool(
        #     tensor34,
        #     [1]+kernel2D+[1],
        #     [1]+stride2D+[1],
        #     pad
        # )

    if sqz:
        output = tf.squeeze(output, -1)
    print(ptype+'pool', output.shape)
    return output


def batch_norm(
    tensor234, training=True, trainable=True, decay=0.9,
    eps=1e-6, scope=''
):
    # with tf.variable_scope(scope):#, reuse=False):
    input_dim = int(tensor234.shape[-1])
    res_dims = list(range(len(tensor234.shape)-1))

    init_zero = constant_init(0.)
    init_one = constant_init(1.)

    gamma = tf.get_variable(
        'gamma',
        [input_dim],
        dtype=FLOAT_tf,
        initializer=init_one,
        trainable=trainable
    )
    beta = tf.get_variable(
        'beta',
        [input_dim],
        dtype=FLOAT_tf,
        initializer=init_zero,
        trainable=trainable
    )

    ema_mean = tf.get_variable(
        'ema_mean', [input_dim],
        dtype=FLOAT_tf,
        initializer=init_zero,
        trainable=False
    )
    ema_var = tf.get_variable(
        'ema_var',
        [input_dim],
        dtype=FLOAT_tf,
        initializer=init_one,
        trainable=False
    )

    if training:
        batch_mean, batch_var = tf.nn.moments(
            tensor234,
            res_dims,
            name='moments'
        )
        train_mean = tf.assign(
            ema_mean, ema_mean * decay + batch_mean * (1.-decay)
        )
        train_var = tf.assign(
            ema_var, ema_var * decay + batch_var * (1.-decay)
        )

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                tensor234,
                tf.identity(batch_mean),
                tf.identity(batch_var),
                beta,
                gamma,
                eps
            )
    else:
        return tf.nn.batch_normalization(
            tensor234,
            ema_mean,
            ema_var,
            beta,
            gamma,
            eps
        )
