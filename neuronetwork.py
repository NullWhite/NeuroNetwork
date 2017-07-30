import tensorflow as tf
import pickle
import math

class neuronetwork:
    def __init__(self):
        self.weights, self.bias = self.__constructNet()
        self.__optimiter()
        self.sess = self.startNet()

    def startNet(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        return sess

    def __constructNet(self):
        '''
        抽象函数,构建网络
        :return: 网络中的参数和偏置字典。
        '''
        raise NotImplementedError()

    def __optimiter(self):
        """
        优化器
        :return:
        """
        raise NotImplementedError()

    def __initWB(self, shape, stddev, value=0.1):
        """
        参数和偏置初始化方法
        :return: 初始化后的参数和偏置
        """
        weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32), dtype=tf.float32)
        bias = tf.Variable(tf.constant(value,dtype=tf.float32, shape=shape[1],), dtype=tf.float32)
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
    def createTNNLayer(self, shape, layerin):
        weight, bias = self.__initWB(shape, math.sqrt(2 / (shape[0] + shape[1])))
        layer = tf.nn.relu(tf.matmul(weight, layerin) + bias)
        return layer, weight, bias

class CNN(neuronetwork):
    def createCNNLayer(self, layerin, layerinshape, coreshape, corenum,strides=[1,1,1,1],padding="SAME"):
        weight, bias = self.__initWB(coreshape + [layerinshape[2], corenum],
                                     math.sqrt(2 / ((coreshape[0] + coreshape[1]) * layerinshape[2])))
        conv1=tf.nn.relu(tf.nn.conv2d(layerin,weight,strides=strides,padding="SAME")+bias)
        return conv1,weight,bias,layerinshape[:-1]+[coreshape]

    def createMaxPool(self,layerin,layerinshape,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME'):
        pool=tf.nn.max_pool(layerin,ksize,strides,padding)
        return pool,[int(layerinshape[0]/strides[1]),int(layerinshape[1]/strides[2])]+layerinshape[2:]