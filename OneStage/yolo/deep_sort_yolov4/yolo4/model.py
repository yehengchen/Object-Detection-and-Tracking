"""YOLO_v4 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.base_layer import Layer
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo4.utils import compose

class Mish(Layer):


    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)
    route = Concatenate()([postconv, shortconv])
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo4_body(inputs, num_anchors, num_classes):
    """Create YOLO_V4 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))

    #19x19 head
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(darknet.output)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(y19)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(y19)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(y19)
    y19 = Concatenate()([maxpool1, maxpool2, maxpool3, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)

    y19_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(y19)

    #38x38 head
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(darknet.layers[204].output)
    y38 = Concatenate()([y38, y19_upsample])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(y38)

    #76x76 head
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(darknet.layers[131].output)
    y76 = Concatenate()([y76, y38_upsample])
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)
    y76 = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76 = DarknetConv2D_BN_Leaky(128, (1,1))(y76)

    #76x76 output
    y76_output = DarknetConv2D_BN_Leaky(256, (3,3))(y76)
    y76_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y76_output)

    #38x38 output
    y76_downsample = ZeroPadding2D(((1,0),(1,0)))(y76)
    y76_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))(y76_downsample)
    y38 = Concatenate()([y76_downsample, y38])
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)
    y38 = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38 = DarknetConv2D_BN_Leaky(256, (1,1))(y38)

    y38_output = DarknetConv2D_BN_Leaky(512, (3,3))(y38)
    y38_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y38_output)

    #19x19 output
    y38_downsample = ZeroPadding2D(((1,0),(1,0)))(y38)
    y38_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(y38_downsample)
    y19 = Concatenate()([y38_downsample, y19])
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)
    y19 = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19 = DarknetConv2D_BN_Leaky(512, (1,1))(y19)

    y19_output = DarknetConv2D_BN_Leaky(1024, (3,3))(y19)
    y19_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(y19_output)

    yolo4_model = Model(inputs, [y19_output, y38_output, y76_output])

    return yolo4_model


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=200,
              score_threshold=.5,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute softmax focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_boxes).
    """

    # Scale predictions so that the class probas of each sample sum to 1
    #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    #epsilon = K.epsilon()
    #y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate Focal Loss
    softmax_focal_loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

    return softmax_focal_loss


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
    #sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

    return sigmoid_focal_loss


def box_iou(b1, b2):
    """
    Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b1, b2):
    """
    Calculate GIoU loss on anchor boxes
    Reference Paper:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        https://arxiv.org/abs/1902.09630

    Parameters
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # get enclosed area
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
    giou = K.expand_dims(giou, -1)

    return giou


def box_diou(b1, b2):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    # calculate param v and alpha to extend to CIoU
    #v = 4*K.square(tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) - tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)
    #alpha = v / (1.0 - iou + v)
    #diou = diou - alpha*v

    diou = K.expand_dims(diou, -1)
    return diou


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo4_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0, use_focal_loss=False, use_focal_obj_loss=False, use_softmax_loss=False, use_giou_loss=False, use_diou_loss=False):
    '''Return yolo4_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        if use_focal_obj_loss:
            # Focal loss for objectness confidence
            confidence_loss = sigmoid_focal_loss(object_mask, raw_pred[...,4:5])
        else:
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask

        if use_focal_loss:
            # Focal loss for classification score
            if use_softmax_loss:
                class_loss = softmax_focal_loss(true_class_probs, raw_pred[...,5:])
            else:
                class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[...,5:])
        else:
            if use_softmax_loss:
                # use softmax style classification output
                class_loss = object_mask * K.expand_dims(K.categorical_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True), axis=-1)
            else:
                # use sigmoid style classification output
                class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)


        if use_giou_loss:
            # Calculate GIoU loss as location loss
            raw_true_box = y_true[l][...,0:4]
            giou = box_giou(pred_box, raw_true_box)
            giou_loss = object_mask * box_loss_scale * (1 - giou)
            giou_loss = K.sum(giou_loss) / mf
            location_loss = giou_loss
        elif use_diou_loss:
            # Calculate DIoU loss as location loss
            raw_true_box = y_true[l][...,0:4]
            diou = box_diou(pred_box, raw_true_box)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            diou_loss = K.sum(diou_loss) / mf
            location_loss = diou_loss
        else:
            # Standard YOLO location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
            xy_loss = K.sum(xy_loss) / mf
            wh_loss = K.sum(wh_loss) / mf
            location_loss = xy_loss + wh_loss

        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += location_loss + confidence_loss + class_loss
        total_location_loss += location_loss
        total_confidence_loss += confidence_loss
        total_class_loss += class_loss

    # Fit for tf 2.0.0 loss shape
    loss = K.expand_dims(loss, axis=-1)

    return loss #, total_location_loss, total_confidence_loss, total_class_loss


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
