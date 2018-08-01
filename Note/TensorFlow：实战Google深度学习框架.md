# TensorFlow：实战Google深度学习框架笔记
## 第二章 环境搭建

### 主要依赖

#### Protocol Buffer
**谷歌开发的处理结构化数据的工具。**
- 序列化之后的数据不是可读的字符串，而是二进制流。
- 使用时需要先定义数据的格式。
- 序列化后的数据比XML格式的数据更小，解析更快。
- message中定义的属性可以是必须的、可选的、可重复的。
#### Bazel
**谷歌开发的自动化构建工具。**

## 第三章 入门

### 计算图

#### 概念

Tensor：张量
- 可以简单的解释为多维数组
- 表面了它的数据结构
Flow：流
- 体现了它的计算模型
- 直观的表达了张量之间通过计算相互转化的过程
TensorFlow
- 是一个通过计算图的形式来表述计算的编程系统
- 每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系

> 在TensorFlow程序中，系统会自动维护一个默认的计算图，通过tf.get_default_graph函数可以获取当前默认的计算图。

### 张量

#### 张量的概念
零阶张量为标量（scalar），也就是一个数。
一阶张量为向量（vector），也就是一个一维数组。
n阶张量为n维数组

张量并不真正保存数字，它保存的是如何得到这些数字的计算过程。

一个张量主要保存了三个属性
- 名字 name（唯一标识符，同时给出这个张量是如何计算出来的） 
- 维度 shape（说明了张量是几维数组，长度是多少）
- 类型 type（每一个张量会有一个唯一的类型，参与运算的所有张量类型需要匹配）

> 通过 node:src_output 的形式给出。
node为节点的名称，src_output表示当前张量来自节点的第几个输出。

#### 张量的使用

第一类用途是对中间计算结果的引用。（提高可读性）
第二类用途是当计算图构造完成之后，张量可以用来获得计算结果。（使用会话session获得结果）

### 运行模型——会话
使用with自动释放致远
```python
import tensorflow as tf
# 创建一个会话，并通过Python中的上下文管理器来管理这个会话
with tf.Session() as sess:
    # 使用这个创建好的会话来计算结果
    sess.run(...)
    # 不需要再调用Session.close()函数来关闭会话
    # 当上下文退出时会话关闭和资源释放也自动完成了    
```
TensorFlow不会自动生成默认的会话，而需要手动指定
当默认的会话被指定之后可以通过tf.Tensor.eval函数来计算一个张量的取值
```python
import tensorflow as tf
sess = tf.Session()
with sess.as_default():
    print(result.eval())

# 以下代码也可以完成相同的功能
sess = tf.Session()
print(sess.run(result))
print(result.eval(session=sess))
```

tf.InteractiveSession函数会自动将生成的会话注册为默认会话.
ConfigProto可以给会话配置类似并行的线程数,GPU分配策略,运算超时时间等参数.

### TensorFlow实现神经网络

#### 神经网络简介
使用神经网络的四个步骤

- 提取问题中实体的特征向量作为神经网络的输入.不同的实体可以提取不同的特征向量.
- 定义神经网络的结构,并定义如何从神经网络的输入得到输出.(神经网络的前向传播算法)
- 通过训练数据来调整神经网络中参数的取值.(训练神经网络的过程)
- 使用训练好的神经网络来预测未知的数据.

#### 前向传播算法简介

神经元是构件一个神经网络的最小单元.
一个神经元有多个输入和一个输出.
每个神经元的输入既可以是其他神经元的输出,也可以是整个神经网络的输入.
所谓神经网络的结构就是指的不同的神经元之间的连接结构.
一个最简单的神经元结构的输出就是所有输入的加权和,而不同输入的权重就是神经元的参数.
神经网络的优化过程就是优化神经元中参数取值的过程.

>**全连接神经网络**: 相邻两层之间任意两个节点之间都有连接.

计算神经网络的前向传播结果需要三部分信息.

1.神经网络的输入,就是从实体中提取的特征向量.

2.神经网络的连接结构.

3.每个神经元中的参数.

![](http://ww1.sinaimg.cn/large/8d8126e8gy1ftf5jt7w2tj20r90hiahb.jpg)

前向传播算法可以表示为矩阵乘法

在TensorFlow中实现
```python
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# tf.matmul实现了矩阵乘法的功能
```

#### 神经网络参数与TensorFlow变量

变量(tf.Variable)的作用是保存和更新神经网络中的参数.

变量需要指定初始值,在神经网络中,给参数赋予随机初始值最为常见.

tf.Variable(tf.random_normal([2, 3], stddev=2))
会产生一个2*3的矩阵,矩阵中的元素均值为0,标准差为2的随机数.
tf.random_normal函数可以通过参数mean来指定平均值,在没有指定时默认为0.

![](http://ww1.sinaimg.cn/large/8d8126e8gy1ftf61dga88j20sk08vaf1.jpg)

![](http://ww1.sinaimg.cn/large/8d8126e8gy1ftf61umivcj20ss06qn1b.jpg)

神经网络中,偏置项(bias)通常会使用常数来设置初始值.

同时,TensorFlow也支持通过其他变量的初始值来初始化新的变量.

样例
```python
import tensorflow as tf
# 声明w1,w2两个变量.这里还通过seed参数设定了随机种子
# 这样可以保证每次运行得到的结果一致.
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量.这里的x是一个1*2的矩阵
x = tf.constant([[0.7, 0.9]])

# 通过前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 因为w1和w2都还没有运行初始化过程.
sess.run(w1.initializer)  # 初始化w1
sess.run(w2.initializer)  # 初始化w2
# 输出[[ 3.95757794]]
print(sess.run(y))
sess.close()
```
TensorFlow提供了一种更加便捷的方式来完成变量初始化过程.
tf.initialize_all_variables()
 
 #### 通过TensorFlow训练神经网络模型
 
 在神经网络优化算法中,最常用的方法是反向传播算法.
 
 首先需要选取一小部分训练数据(batch).
 然后这个Batch的样例会通过前向传播算法得到神经网络模型的预测结果.
 因为训练数据都是有正确答案标注的,
 所以可以计算出当前神经网络模型的预测答案与正确答案之间的差距.
 最后,基于这个预测值和真实值之间的差距,
 反向传播算法会相应更新神经网络参数的取值,
 使得在这个Batch上神经网络模型的预测结果和真实值答案更加接近.
 
 为了提高利用率和减少计算图大小,
 TensorFlow提供了placeholder机制用语提供输入数据.
 placeholder相当于定义了一个位置,
 这个位置中的数据在程序运行时再指定.
 
 下面通过placeholder实现前向传播算法
 ```python
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方(维度不一定要定义)
# 如果维度是确定的,那么给出维度可以降低出错的概率.

x  = tf.placeholder(tf.float32([2, 3], shape=(1, 2), name='input'))
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
```

一次性计算多个样例的前向传播结果
```python
x  = tf.placeholder(tf.float32([2, 3], shape=(3, 2), name='input'))
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
```

得到一个Batch的前向传播结果之后,
需要定义一个损失函数来刻画当前的预测值和真是答案之间的差距.
然后通过反向传播算法来调整神经网络参数的取值
使得差距可以被缩小.

一个简单的损失函数,并通过TensorFlow定义了反向传播的算法

```python
# 定义损失函数来刻画预测值和真实值的差距(交叉熵)
cross_entropy = - tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
)
# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
```

> 目前TensorFlow支持7种不同的反向传播优化器.常用的优化方法有三种
tf.train.GradientDescentOptimizer
tf.train.AdamOptimizer
tf.train.MomentumOptimizer

#### 完整神经网络样例程序

```python
import tensorflow as tf

# NumPy是一个科学计算的工具包,这里通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练数据Batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None可以方便使用不大的Batch大小.
# 在训练时需要把数据分成比较小的Batch,但在测试时,可以一次使用全部数据.

x = tf.placeholder(tf.placeholder(tf.float32, shape=(None, 2), name='x-input'))
y_ = tf.placeholder(tf.placeholder(tf.float32, shape=(None, 1), name='y-input'))

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
cross_entropy = - tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
)
# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 正样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    # 初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    
    # 设定训练的轮数
    Steps = 5000
    for i in range(Steps):
        # 每次选取batch_size个样本进行训练.
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y}
            )
            print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))
    print(sess.run(w1))  
    print(sess.run(w2))
    
```

## 第四章 深层神经网络

### 深度学习与深度神经网络

深度学习
> **定义**: 一类通过多层非线性变换对高复杂性数据建模算法的合集

因为深层神经网络是实现"多层非线性变换"最常用的一种方法.
所以基本上可以认为深度学习就是深度神经网络的代名词.
特性  多层和非线性

#### 线性模型的局限性

当线性模型的输入只有一个时,x和y形成了二维坐标系上的一条直线.

当线性模型的输入有n个时,x和y形成了n+1维空间中的一个平面.

而一个线性模型中通过输入得到输出的函数被称为一个线性变换.

线性模型的最大特点是任意线性模型的组合仍然是线性模型.

只通过线性变换,任意层的全连接神经网络和单层的神经网络模型的表达能力没有任何区别.

#### 激活函数实现去线性化

如果将每一个神经元(神经网络中的节点)的输出通过一个非线性函数,那么整个神经网络的模型也就不再是线性的了.

加上激活函数和偏置项后的前向传播算法与之前的变化:

- 增加了偏置项(bias)
- 每个节点的取值不再是单纯的加权和(在加权和的基础上还做了一个非线性变换)

![](http://ww1.sinaimg.cn/large/8d8126e8gy1ftfbtgxd8bj20rv0btadp.jpg)

TensorFlow支持7种非线性激活函数,也支持使用自己定义的激活函数

```python
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)
```

#### 多层网络解决异或运算

深度神经网络实际上有组合特征提取的功能.

这个特征对于解决不易提取特征向量的问题(图片识别,语音识别等)有很大帮助.

### 损失函数定义

#### 经典损失函数

##### 分类问题

判断一个输出向量和期望向量的接近程度用**交叉熵**.
交叉熵刻画了两个概率分布之间的距离,它是分类问题中使用比较广的损失函数.

交叉熵公式
![|center](http://ww1.sinaimg.cn/large/8d8126e8gy1ftg3gsq9jvj208901sdfy.jpg)
> 这里的p是正确样例的概率分布,q是预测样例的概率分布

tf.clip_by_value函数可以将一个张量中的数值限制在一个范围之内.
这样可以避免一些运算错误(比如log0是无效的).样例
```python
v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(tf.clip_by_value(v, 2.5, 4.5).eval())
# 输出[[2.5, 2.5, 3.0], [4.0, 4.5, 4.5]]
```

#### 自定义损失函数

```python
loss = tf.reduce_sum(tf.select(tf.greater(v1, v2), (v1 - v2) * a, (v2 - v1) * b))
```
tf.greater的输入是两个张量,此函数会比较这两个张量中每一个元素的大小,并返回比较结果.

tf.select函数有三个参数,第一个为选择条件根据,当选择条件为True时,选择第二个参数中的值,否则使用第三个参数中的值.

### 神经网络优化算法

- 反向传播算法(backpropagation)

- 梯度下降算法(gradient decent)

梯度下降算法主要用于优化单个参数的取值,
而反向传播算法给出了一个高效的方式在所有参数是上使用梯度下降算法,
从而使神经网络模型在训练数据上的损失函数尽可能小.
反向传播算法是训练神经网络的核心算法,
它可以根据定义好的损失函数优化神经网络中的参数的取值,
从而使神经网络模型在训练数据集上的损失函数达到一个较小值.
神经网络模型中参数的优化过程直接决定了模型的质量.

梯度下降算法不能保证被优化的函数达到全局最优解.
只有当损失函数为凸函数时,梯度下降算法才能保证达到全局最优解.

梯度下降算法的问题在于计算时间太长,为了加速训练过程,可以使用随机梯度下降的算法.

实际应用中一般每次计算一小部分训练数据的损失函数.这一小部分数据被称为一个batch.
在一个batch上优化神经网络的参数并不会比单个数据慢太多,
而且可以大大减小所需迭代次数,
同时可以使收敛的结果更加接近梯度下降的效果.

### 神经网络进一步优化

> 通过指数衰减的方法设置梯度下降算法中的学习率.
通过指数衰减的学习率既可以让模型在训练的前期快速接近较优解,
又可以保证模型在训练后期不会有太大的波动,从而更加接近局部最优.

#### 学习率的设置

在训练神经网络时,需要设置学习率(learning rate)空值参数更新的速度.

TensorFlow提供了一种更加灵活的学习率设置方法——指数衰减法.
tf.train.exponential_decay函数实现了指数衰减学习率.

```python
global_step = tf.Variable(0)

# 通过exponential_decay函数生成学习率
learning_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True
)
# 使用指数衰减的学习率.在minimize函数中传入global_step将自动更新
# global_step参数,从而使学习率也得到更新
learning_step = tf.trian.GradientDescentOptimizer(learning_rate).minimize(...my loss..., global_step=global_step)
```

#### 过拟合问题

为了避免过拟合,通常使用正则化(regularization).

正则化的思想就是在损失函数中加入刻画模型复杂程度的指标.

基本思想都是通过限制权重的大小,使得模型不能任意你和训练数据中的随机噪音.

带L2正则化的损失函数
```python
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)

loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l2_regularizer(lambda)(w))
```

神经网络参数增多会导致损失函数loss的定义很长,可读性变差.
使用集合来计算一个5层神经网络待L2正则化的损失函数的计算方法.
```python
import tensorflow as tf

# 获取一层神经网络边上的权重,并将这个权重的L2正则化损失加入名称为losses的集合中
def get_weight(shape, score):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection函数将这个新生成变量的L2正则化损失项加入集合
    # 这个函数的第一个参数'losses'是集合的名字,第二参数是要加入这个集合的内容.
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(score)(var)
    )
    # 返回生成的变量
    return var
  
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点,开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量,并将这个变量的L2正则化损失加入计算图上的集合.
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同事已经将所有L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上的表现的损失函数.
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection返回一个列表,这个列表是所有这个集合中的元素.
# 在这个样例中,这些元素就是损失函数的不同部分,将他们加起来就可以得到最终的损失函数.
loss = tf.add_n(tf.get_collection('losses'))
```


#### 滑动平均模型

可以使模型在测试数据上更健壮的方法.
TensorFlow中提供了tf.train.ExponentialMovingAverage来实现滑动平均模型.
在初始化ExponentialMovingAverage时,需要提供一个衰减率(decay)用于控制模型更新速度.
ExponentialMovingAverage对每一个变量会维护一个影子变量(shadow variable),
这个影子变量的初始值就是相应变量的初始值,而每次运行变量更新时,影子变量的值会更新为:
![|center](http://ww1.sinaimg.cn/large/8d8126e8gy1ftiebl3l8ij20jn00yt96.jpg)
其中shadow_variable为影子变量,variable为待更新的变量,decay为衰减率.
decay决定了模型的速度,decay越大模型越趋于稳定.
在实际应用中,decay一般会设置为非常接近1的数,
样例
```python
import tensorflow as tf
# 定义一个变量用于计算滑动平均,这个变量的初始值为0
v1 = tf.Variable(0, dtype=tf.float32)
# step变量模拟神经网络中迭代的轮数,可以用于动态控制衰减率.
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类,初始化时给定了衰减率(0.99)和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作.
# 这里需要给定一个列表,每次执行这个操作时,这个列表中的变量都会被更新.
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    
    # 通过ema.average(v1)获取滑动平均水平之后变量的取值.
    # 在初始化之后变量v1的值和v1的滑动平均都为0
    print(sess.run([v1, ema.average(v1)]))
    
    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值.衰减率为min{0.99, (1+step)/(10/step) =0.1}=0.1
    # 所以v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    
    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值.衰减率为min{0.99, (1+step)/(10/+step)0.999}=0.99
    # 所有v1的滑动平均会被更新为0.99*4.5+0.01*10=4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    # 输出[10.0, 4.5549998]
    
    # 再次更新滑动平均值,得到的新滑动平均值为0.99*4.555+0.01*10=4.60945
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
    # 输出[10.0, 4.6094499
```


## 第五章 MNIST数字识别问题


### MNIST数据处理
```
```