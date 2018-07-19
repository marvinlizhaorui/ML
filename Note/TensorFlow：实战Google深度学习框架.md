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
```
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
```
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
```
x  = tf.placeholder(tf.float32([2, 3], shape=(3, 2), name='input'))
print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
```

得到一个Batch的前向传播结果之后,
需要定义一个损失函数来刻画当前的预测值和真是答案之间的差距.
然后通过反向传播算法来调整神经网络参数的取值
使得差距可以被缩小.