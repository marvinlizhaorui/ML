# -*- coding: utf-8 -*-
# @Time    : 18-7-23 上午11:28
# @Author  : Marvin
# @File    : MNIST-5.1.py
# @Notes   : 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/marvin/Documents/py3_wts/ML/ML/mnist/', one_hot=True)

# 训练数据的数量
print('Training data size: ', mnist.train.num_examples)
# 验证数据的数量
print('Validating data size: ', mnist.validation.num_examples)
# 测试数据的数量
print('Testing data size: ', mnist.test.num_examples)

print('Example training data: ', mnist.train.images[0])
print('Example training data label: ', mnist.train.labels[0])

# 处理后每一张图片是一个长度为784(28*28)的一维数组

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train的集合中选取batch_size个训练数据
print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)

