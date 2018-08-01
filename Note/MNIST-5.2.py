# -*- coding: utf-8 -*-
# @Time    : 18-7-23 上午11:44
# @Author  : Marvin
# @File    : MNIST-5.2.py
# @Notes   : 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关的常数
input_node = 784  # 输入层的节点数
output_node = 10  # 输出层的节点数

# 配置神经网络的参数

layer1_node = 500  # 隐藏层节点数
batch_size = 100  # 一个训练batch中的训练数据个数.

learning_rate_base = 0.8  # 基础学习率
learning_rate_decay = 0.99  # 学习率的衰减率

regularization_rate = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
training_steps = 30000  # 训练轮数
moving_average_decay = 0.99  # 滑动平均衰减率


# 一个辅助函数,给定神经网络的输入和所有参数,计算神经网络的前向传播结果.
# 在这里定义了一个使用ReLU激活函数的三层全连接神经网络.
# 通过加入隐藏层实现了多层网络结构,通过ReLU激活函数实现了去线性化.
# 在这个函数中也支持传入用于计算参数平均值的类,这样方便在测试时使用滑动平均模型.


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时,值使用参数当前的取值
    if avg_class is None:
        # 计算隐藏层的前向传播结果,这里使用了ReLU激活函数.
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 计算输出层的前向传播结果.因为在计算损失函数时会一并计算softmax函数.
        # 所以这里不需要加入激活函数.而且不加入softmax不会影响预测结果.
        # 因为预测时使用的是不同类别对应节点输出值的相对大小,有没有softmax层对最后分类结果的计算没有影响.
        # 于是在计算整个神经网络的前向传播时可以不加入最后的softma层.
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播结果.
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )

        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([input_node, layer1_node], stddev=0.1)
    )

    biases1 = tf.Variable(
        tf.constant(0.1, shape=[layer1_node])
    )

    # 生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([layer1_node, output_node], stddev=0.1)
    )

    biases2 = tf.Variable(
        tf.constant(0.1, shape=[output_node])
    )

    # 计算在当前参数下神经网络前向传播的结果.这里给出的用语计算滑动平均的类为None,所以函数不会使用参数的滑动平均值.
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量.这个变量不需要计算滑动平均值,所以这里指定这个变量为不可训练的变量(trainable=False)
    # 在使用Tensor训练神经网络时,一般会将代表训练轮数的变量指定为不可训练的参数.
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量,初始化滑动平均类.
    # 给定训练论数的变量可以加快训练早期变量的更新速度.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step
    )

    # 在所有代表神经网络参数的变量上使用滑动平均.其他辅助变量就不需要了.
    # tf.trainable_variables返回的就是图上集合
    # GraphKeys.TRAINABLE_VARIABLES中的元素.这个集合的元素就是所有没有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    # 计算使用了滑动平均之后的前向传播结果。
    average_y = inference(
        input_tensor=x, avg_class=variable_averages, weights1=weights1, weights2=weights2, biases1=biases1, biases2=biases2
    )

    # 计算交叉熵作为刻画预测值和真实值之间的差距的损失函数。
    # 这里TensorFlow中提供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵.
    # 当分类问题只有一个正确答案时,可以使用这个函数来加速交叉熵的计算.
    # 第一个参数是神经网络不包括softmax层的前向传播结果
    # 第二个参数是训练数据的正确答案,使用tf.argmax函数来获得正确答案对应的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        y, tf.argmax(y_, 1)
    )

    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    # 计算模型的正则化损失.一般只计算神经网络边上权重的正则化损失,而不使用偏置项.
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失的和.
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,  # 基础的学习率,随着迭代的进行,更新变量时使用的
        global_step,  # 当前迭代的轮数
        mnist.train.num_example / batch_size,  # 过完所有的训练数据需要的迭代次数

        learning_rate_decay  # 学习率衰减速度
    )

    # 使用tf.train.GrandientDescentOptimizer优化算法来优化损失函数(包含交叉熵损失和L2正则化损失)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)

    # 在训练神经网络模型时,每过一遍数据既需要通过反向传播来更新神经网络中的参数
    # 又要更新每一个参数的滑动平均值.
    # TensorFlow提供两种机制, tf.group和tf.control_dependencies
    # train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确.
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 这个运算首先将一个布尔型的数值转换为实数型,然后计算平均值,平均值就是模型的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {
            x: mnist.validation.image,
            y_: mnist.validation.labels
        }

        test_feed = {
            x: mnist.test.image,
            y_: mnist.test.labels
        }

        # 迭代地训练神经网络
        for i in range(training_steps):
            # 每1000轮输出一次在验证数据集上的测试结果.
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy"
                      "using average model is %g " % (i, validate_acc))

                # 产生这一轮使用的一个batch的训练数据,并运行训练过程
                xs, ys = mnist.train.next_batch(batch_size)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后,在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average '
              "model is %g " % (training_steps, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('/home/marvin/Documents/py3_wts/ML/ML/mnist/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
