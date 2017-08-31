import tensorflow as tf
import pickle
import math


class neuronetwork:
    def __init__(self):
        self.weights, self.bias = self.constructNet()
        self.optimiter()
        self.sess = self.startNet()

    def startNet(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        return sess

    def constructNet(self):
        '''
        抽象函数,构建网络
        :return: 网络中的参数和偏置字典。
        '''
        raise NotImplementedError()

    def optimiter(self):
        """
        优化器
        :return:
        """
        raise NotImplementedError()

    def initWB(self, shape, stddev, value=0):
        """
        参数和偏置初始化方法
        :return: 初始化后的参数和偏置
        """
        weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32), dtype=tf.float32)
        bias = tf.Variable(tf.constant(value,dtype=tf.float32, shape=[shape[-1]]), dtype=tf.float32)
        return weight, bias

    def queryWB(self):
        """
        查询网络的参数和偏置
        :return:
        """
        weights = {}
        bias = {}
        for item in self.weights:
            weights[item] = self.sess.run(self.weights[item])
        for item in self.bias:
            bias[item] = self.sess.run(self.bias[item])
        return weights, bias

    def saveWB(self, filepath):
        """
        保存参数和偏置
        :param filepath:保存的文件地址
        :return:
        """
        weights, bias = self.queryWB()
        file = open(filepath, 'wb')
        pickle.dump((weights, bias), file)
        file.close()

    def loadWB(self, filepath):
        """
        从文件导入参数和偏置
        :param filepath:导入源文件的地址
        :return:参数、偏置
        """
        file = open(filepath, 'rb')
        weights, bias = pickle.load(file)
        return weights, bias

    def train(self, data_batch, label_batch):
        """
        训练网络
        :param data_batch: 输入数据块
        :param label_batch: 标签数据块
        :return: 训练的代价
        """
        raise NotImplementedError

    def calPercision(self, data_batch, label_batch):
        """
        计算准确度
        :param data_batch: 计算数据块
        :param label_batch: 验证数据标签块
        :return: 精度
        """
        raise NotImplementedError

    def calResult(self, data_batch):
        """
        计算结果
        :param data_batch: 需要计算的数据块
        :return: 计算结果
        """
        raise NotImplementedError

class TNN(neuronetwork):
    def __init__(self):
        neuronetwork.__init__(self)

    def createTNNLayer(self, shape, layerin,sddev=None,value=None,):
        if sddev==None:
            sddev=math.sqrt(2 / (shape[0] + shape[1]))
        if value==None:
            value=0
        weight, bias = self.initWB(shape, sddev, value)
        layer = tf.nn.relu(tf.matmul(layerin, weight) + bias)
        return layer, weight, bias

    def createSMLayer(self,shape,layerin):
        weight, bias = self.initWB(shape, math.sqrt(2 / (shape[0] + shape[1])))
        layer=tf.nn.softmax(tf.matmul(layerin,weight)+bias)
        return layer,weight,bias

class CNN(TNN):
    def __init__(self):
        TNN.__init__(self)

    def createCNNLayer(self, layerin, layerinshape, coreshape, corenum,sddev=None,value=None,strides=[1,1,1,1],padding="SAME"):
        if sddev==None:
            sddev=math.sqrt(2 / ((coreshape[0] * coreshape[1]) * corenum))
        if value==None:
            value=0
        weight, bias = self.initWB(coreshape + [layerinshape[2], corenum],sddev,value)
        conv1=tf.nn.relu(tf.nn.conv2d(layerin,weight,strides=strides,padding=padding)+bias)
        return conv1, weight, bias, [item.value for item in conv1.shape.dims][1:]

    def createMaxPool(self,layerin,layerinshape,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
        pool=tf.nn.max_pool(layerin,ksize,strides,padding)
        return pool, [item.value for item in pool.shape.dims][1:]
