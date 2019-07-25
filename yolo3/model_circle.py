# -*- coding: utf-8 -*-
# @Time    : 19-7-4 上午11:17
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : model_circle.py
# @Software: PyCharm
import cv2
import numpy as np
from functools import reduce
from keras import backend as K
import tensorflow as tf
import math


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True, max_circle=4,
                    alpha=2, rotateAgle=15, proc_img=True):
    '''随机处理图像并进行数据增广'''

    line = annotation_line.split()
    image = cv2.imread(line[0])
    iw, ih = image.shape[1], image.shape[0]
    h, w = input_shape
    circle = np.array([np.array(list(map(int, circle.split(',')))) for circle in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        resize_w = int(scale * iw)
        resize_h = int(scale * ih)
        padding_w = (w - resize_w) // 2
        padding_h = (h - resize_h) // 2
        image_data = 0

        if proc_img:
            image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
            image = cv2.copyMakeBorder(image, padding_h, padding_h, padding_w, padding_w, cv2.BORDER_CONSTANT,
                                       value=(128, 128, 128))
            image_data = np.array(image) / 255.

        # correct circle
        circle_data = np.zeros((max_circle, 4))
        if (len(circle) > 0):
            np.random.shuffle(circle)
            if len(circle) > max_circle:
                circle = circle[:max_circle]  # 若lable太多 只取max_circle个
            circle[:, 0] = circle[:, 0] * scale + padding_w
            circle[:, 1] = circle[:, 1] * scale + padding_h
            circle[:, 2] = circle[:, 2] * scale
            circle_data[:len(circle)] = circle

        return image_data, circle_data

    # 实现灰度图像亮度调节,label坐标和半径不变
    alpha = rand(.5, alpha)
    blank = np.zeros(image.shape, image.dtype)
    image = cv2.addWeighted(image, alpha, blank, 1 - alpha, 0)

    # 实现图像尺寸变化,label坐标和半径都需要resize一个尺寸
    scale = rand(.1, .5)
    resize_w, resize_h = int(iw * scale), int(ih * scale)
    image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

    # 对图片进行随机旋转操作,指定旋转的中心为图片中心,label坐标需要旋转,半径不变,
    rotateCenter = (image.shape[0] / 2, image.shape[1] / 2)
    rotateAgle = rand(-rotateAgle, rotateAgle)
    rotateMatrix = cv2.getRotationMatrix2D(rotateCenter, rotateAgle, 1)
    image = cv2.warpAffine(image, rotateMatrix, input_shape, flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    image_data = np.array(image) / 255.

    # correct circle
    circle_data = np.zeros((max_circle, 4))
    if (len(circle) > 0):
        np.random.shuffle(circle)
        if len(circle) > max_circle:
            circle = circle[:max_circle]  # 若lable太多 只取max_circle个

        # 尺度变化
        circle[:, :3] = circle[:, :3] * scale

        # 旋转
        circle_Rotate = np.transpose(circle.copy())[:3, :]
        circle_Rotate[2, :] = 1
        circle_Rotate = np.transpose(rotateMatrix.dot(circle_Rotate))
        circle[:, :2] = circle_Rotate[:, :2]

        circle_data[:len(circle)] = circle

    # for i in range(len(circle_data)):
    #     cv2.circle(image, (int(circle_data[i, 0]), int(circle_data[i, 1])), int(circle_data[i, 2]), (255, 0, 0), 1)
    # cv2.imshow('process', image)
    # cv2.waitKey()

    return image_data, circle_data


def compute_incross_iou(pre_c_r, ture_c_r, distance):
    alpha1 = tf.acos((pre_c_r ** 2 + distance ** 2 - ture_c_r ** 2) / (distance * pre_c_r * 2))
    s1 = alpha1 * pre_c_r * 2
    alpha2 = tf.acos((ture_c_r ** 2 + distance ** 2 - pre_c_r ** 2) / (distance * ture_c_r * 2))
    s2 = alpha2 * ture_c_r * 2
    s = distance * pre_c_r * tf.sin(alpha1)
    intersect_s = s1 + s2 - s
    s11 = math.pi * ture_c_r ** 2
    s22 = math.pi * pre_c_r ** 2
    incross_iou = intersect_s / (s11 + s22 - intersect_s)
    return incross_iou


def yolo_correct_circle(circle_xy, circle_r, input_shape, image_shape):
    """

    :param circle_xy:
    :param circle_r:
    :param input_shape:
    :param image_shape:
    :return: get corrected circle
    """
    circle_xy = circle_xy[..., ::-1]
    input_shape = np.cast(input_shape, np.dtype(circle_xy))
    image_shape = np.cast(image_shape, np.dtype(circle_xy))
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    return new_shape


def yolo_circle_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    circle_xy, circle_r, circle_confidence, circle_class_prob = yolo_head(feats, anchors, num_classes, input_shape)


def circle_iou(pre_c, ture_c):
    pre_c = K.expand_dims(pre_c, -2)  # 在倒数第二行扩充一维
    pre_c_xy = pre_c[..., :2]
    pre_c_r = pre_c[..., 2]

    ture_c = K.expand_dims(ture_c, -2)  # 在前面扩充一维
    ture_c_xy = ture_c[..., :2]
    ture_c_r = ture_c[..., 2]

    '''小半径<1/2大半径,flag=0'''
    max_r = K.maximum(pre_c_r, ture_c_r)
    min_r = K.minimum(pre_c_r, ture_c_r)
    circle_flag = tf.cast(K.greater(min_r, 0.5 * max_r), dtype=tf.float64)  # greater 是 >   less <
    add_r = pre_c_r + ture_c_r
    sub_r = abs(pre_c_r - ture_c_r)

    distance = K.sqrt((pre_c_xy[..., 0] - ture_c_xy[..., 0]) ** 2 + (pre_c_xy[..., 1] - ture_c_xy[..., 1]) ** 2)
    padding = tf.zeros(shape=ture_c_r.shape, dtype=tf.float64)

    '''0=<distance<sub_r;'''
    contain = tf.cast(K.less(distance, sub_r), dtype=tf.float64)
    contain_flag = tf.cast(circle_flag * contain, dtype=tf.bool)
    contain_iou = min_r ** 2 / max_r ** 2
    contain_iou = tf.where(contain_flag, contain_iou, padding)

    '''sub_r<=distance<add_r;'''
    incross_large = K.greater(distance, sub_r)
    incross_small = K.less(distance, add_r)

    incross_flag = tf.cast(incross_large, dtype=tf.float64) * tf.cast(incross_small, dtype=tf.float64)
    incross_flag = tf.cast(circle_flag * incross_flag, dtype=tf.bool)
    incross_iou = compute_incross_iou(pre_c_r, ture_c_r, distance)
    incross_iou = tf.where(incross_flag, incross_iou, padding)

    '''add_r=<distance'''
    separation_flag = tf.cast(K.greater(distance, add_r), dtype=tf.float64)
    separation_flag = tf.cast(circle_flag * separation_flag, dtype=tf.bool)
    separation_iou = tf.where(separation_flag, padding, padding)

    iou = incross_iou + contain_iou + separation_iou

    return iou


def yolo_head(feats, anchors, num_class=1, input_shape=(24, 32), calc_loss=False):
    '''convert the final layer features to circle parameters'''
    num_anchor = len(anchors)
    '''reshape to batch,height,width,num_anchors,circle_params'''
    anchors_tensor = K.reshape(K.constant(anchors), shape=[1, 1, 1, num_anchor, 1])
    grid_shape = K.shape(feats)[1:3]  # 原始图片输入的是wh
    grid_y = K.tile(K.reshape(K.arange(start=0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(start=0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, shape=[-1, grid_shape[0], grid_shape[1], num_anchor, num_class + 4])

    circle_xy = (K.sigmoid(feats[..., :2] + grid) / K.cast(grid_shape[::-1], K.dtype(feats)))  # xy
    circle_r = (K.exp(feats[..., 2] * anchors_tensor) / K.cast(input_shape[::-1], K.dtype(feats)))  # r
    circle_confidence = (K.sigmoid(feats[..., 3]))
    circle_probs = K.sigmoid(feats[..., 4])

    if calc_loss == True:
        return grid, feats, circle_xy, circle_r
    return circle_xy, circle_r, circle_confidence, circle_probs


def preprocess_true_circle(true_circle, input_shape, anchors, num_classes):
    '''preprocess true circle to training input format '''
    assert (true_circle[..., 3] < num_classes).all()
    num_layers = len(anchors) // 3
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    true_circle = np.array(true_circle, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)
    circle_xy = true_circle[..., 0:2]
    circle_r = true_circle[..., 2:3]

    assert (input_shape[0] == input_shape[1])

    true_circle[..., 0:2] = circle_xy / input_shape[::-1]
    # true_circle[..., 2:3] = circle_r / input_shape[0]

    m = true_circle.shape[0]
    grid_shape = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shape[l][0], grid_shape[l][1], len(anchors_mask[l]), 4 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    print(y_true[1].shape)

    anchors = np.expand_dims(anchors, -1)

    # print('anchor_shape:', anchors.shape)

    # circle_area = circle_r ** 2
    # anchors_area = anchors ** 2
    # iou = (circle_r / anchors) ** 2
    # anchors = np.reshape(len(anchors), 1)
    # anchors_maxes = anchors
    # anchors_mins = -anchors_maxes

    valid_mask = circle_r[..., 0] > 0
    ious = []
    # print(len(valid_mask))

    # 对于每个样本找到该样本最合适的anchor,即所有anchor中找到和true_circle半径最近的anchor;
    for b in range(m):
        r = circle_r[b, valid_mask[b]]  # 选取有效的label
        if len(r) == 0: continue
        iou = np.abs((r / anchors) ** 2 - 1)

        # print(iou)
        best_anchor = np.argmin(iou, axis=0)
        # print(best_anchor)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchors_mask[l]:
                    print('in')
                    i = np.floor(true_circle[b, 0] * grid_shape[l][1]).astype('int32')
                    j = np.floor(true_circle[b, 1] * grid_shape[l][0]).astype('int32')
                    print(i, j)
                    k = anchors_mask[l].index(n)
                    c = true_circle[b, 3].astype('int32')
                    print(true_circle[b, 0:3])
                    y_true[l][b, j, i, k, 0:3] = true_circle[b, 0:3]
                    y_true[l][b, j, i, k, 3] = 1
                    y_true[l][b, j, i, k, 4 + c] = 1
    return y_true


def yolo_circle_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """
    :brife :computer the loss
    :param args:
        yolo_outputs:the net computer
        y_true:      the output of yolo body or tiny yolo
    :param anchors: array,shape=(N,1) radius
    :param num_classes: integer
    :param ignore_thresh: float,the iou threshold whether to ignore object confidence loss
    :param print_loss:
    :return: loss: tensor,shape=(1,)

    """

    num_layers = len(anchors) // 3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shape = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 3:4]
        ture_class_probs = y_true[l][..., 4:]

        grid, raw_pred, pred_xy, pred_r = yolo_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    calc_loss=True)
        pred_circle = K.concatenate([pred_xy, pred_r])

        raw_true_xy = y_true[l][..., :2] * grid_shape[l][::-1] - grid
        raw_true_r = K.log(y_true[l][..., 2:3] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_r = K.switch(object_mask, raw_true_r, K.zeros_like(raw_true_r))
        circle_loss_scale = 2 - y_true[l][..., 2:3] ** 2

        # find ingnore mask,iterate over each of batch
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_circle = tf.boolean_mask(y_true[l][b, ..., 0:3], object_mask_bool[b, ..., 0])
            iou = circle_iou(pred_circle[b], true_circle)
            best_iou = K.max(iou, axis=1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_circle)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * circle_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                          from_logits=True)
        r_loss = object_mask * circle_loss_scale * 0.5 * K.square(raw_true_r - raw_pred[..., 2:3])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 3:4], from_logits=True) + (
                1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 3:4],
                                                         from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(ture_class_probs, raw_pred[..., 4:], from_logits=True)
        xy_loss = K.sum(xy_loss) / mf
        r_loss = K.sum(r_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + r_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, r_loss, confidence_loss, class_loss, K.sum(ignore_mask)])
    return loss


if __name__ == '__main__':
    true_circle = np.array([[218, 406, 41, 0],
                            [359, 406, 40, 0]])
    input_shape = [416, 416]
    image_shape = [1280, 1024]
    anchors = [10, 13, 11, 21, 20, 22, 32, 31, 30]
    num_classes = 1
    circle_xy = [[218, 406], [359, 406]]
    circle_r = [[41], [40]]
    circle_xy = np.array(circle_xy)
    circle_r = np.array(circle_r)
    print(circle_xy.shape)
    print(circle_r.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # grid_shape = preprocess_true_circle(true_circle, input_shape, anchors, num_classes)
        print(yolo_correct_circle(circle_xy, circle_r, input_shape, image_shape))
