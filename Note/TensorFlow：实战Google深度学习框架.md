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
