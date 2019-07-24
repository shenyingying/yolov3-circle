# -*- coding: utf-8 -*-
# @Time    : 19-7-9 下午4:34
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : test.py
# @Software: PyCharm

# import pygame
# import sys
# import random
# basket_x = 0
# basket_y = 600
# ball_x = 10
# ball_y = 10
# screen_width = 1000
# screen_heigh = 800
# score = 0
#
# pygame.init()
# screen = pygame.display.set_mode((screen_width, screen_heigh))
# pygame.display.set_caption('接球')
# basket = pygame.image.load('/home/sy/timg.jpeg').convert()
# basket_w, basket_h = basket.get_size()
# ball = pygame.image.load('/home/sy/qiu.jpg').convert()
# ball_w, ball_h = ball.get_size()
# while True:
#     pygame.display.update()

import numpy as np


def compBestIou(radius, anchors):
    # iou = radius.dot(anchors)
    iou = radius * anchors
    return iou


if __name__ == '__main__':
    radius = 23
    radius = np.array(radius)
    anchors = [[[10], [11], [12]], [[13], [15], [17]], [[20], [25], [18]]]
    anchors = np.array(anchors)
    print(anchors.shape)
    # print(radius/anchors[..., 0])
    # anchors = np.expand_dims(anchors, -1)  # (10,1)
    iou = compBestIou(radius, anchors)
    print(iou.shape)
    best_anchor = np.argmax(iou, axis=1)
    print(best_anchor)
    # best_anchor=np.array(best_anchor)
    for t,n in enumerate(best_anchor):
        print(anchors[n])
