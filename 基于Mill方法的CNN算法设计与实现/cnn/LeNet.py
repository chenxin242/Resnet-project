import tensorflow as tf

# 定义LeNet模型
# 图片模型


def LeNet(x):
    #  # c1:卷积层 输入 = 32x32x1. 输出 = 28x28x6
    # 1.tf.Variable 创建变量
    # 2.tf.truncated_normal 截断的产生正态分布的随机数，
    #   即随机数与均值的差值若大于两倍的标准差，则重新生成
    #   参数：shape 生成张量的维度  mean: 均值 stddev 标准差
    #         shape：5*5的矩阵(卷积核大小) 厚度1  6个卷积核
    # conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=m, stddev=sigma))
    # 3.tf.zeros(6) 6行1列的0 作为b
    # conv1_b = tf.Variable(tf.zeros(6))
    # 4.tf.nn.conv2d(input,filter,strides,padding) 卷积计算
    #   参数：input 输入图片 filter 卷积核
    #         strides 步长[1, 长上步长，宽上步长，1]
    #         padding 卷积核在边缘处的处理方法 valid(跳过) 和 same(补充)
    # conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # 5.tf.nn.relu()函数激活conv1
    # conv1 = tf.nn.relu(conv1)

    #  # S2:池化层 输入 = 28X28X6 输出 = 14X14X6
    # 1.tf.nn.max_pool(value,ksize,strides,padding) 最大池化
    #   参数：value 输入
    #         ksize [batch,height,width,channels] 池化窗口大小为2X2
    #         strides 步长[1, 长上步长，宽上步长，1]
    #         padding 池化窗口在边缘处的处理方法
    # pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  # C3:卷积层. 输入 = 14X14X6 输出10X10X6
    # 1.创建第二个卷积核 卷积核长5 宽5 厚度6 16个通道
    # conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=m, stddev=sigma))
    # 2.创建16个通道的b
    # conv2_b = tf.Variable(tf.zeros(16))
    # 3.卷积操作
    # conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # 4.relu 函数激活
    # conv2 = tf.nn.relu(conv2)

    #  # S4:池化层 输入 = 10X10X16 输出 = 5X5X16
    # pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # 压缩成1维 输入5X5X16 输出 = 1X400
    # fc1 = flatten(pool_2)

    # # C5:全连接层 输入=400 输出120
    # fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=m, stddev=sigma))
    # fc1_b = tf.Variable(tf.zeros(120))
    # tf.matmul 矩阵相乘
    # fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    # fc1 = tf.nn.relu(fc1)

    # # F6:全连接层 输入=120 输出84
    # fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=m, stddev=sigma))
    # fc2_b = tf.Variable(tf.zeros(84))
    # fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # fc2 = tf.nn.relu(fc2)

    # 输出层 输入=84 输出=10
    # fc3_w = tf.Variable(tf.truncated_normal(shape=(84,10), mean=m, stddev=sigma))
    # fc3_b = tf.Variable(tf.zeros(10))
    # logits = tf.matmul(fc2, fc3_w) + fc3_b
    # return logits
    return True
