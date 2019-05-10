import numpy as np, matplotlib.pyplot as mp
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import (
    Input, Dense, Reshape, Flatten, Embedding,
    Conv2DTranspose, Conv2D, LeakyReLU,
    BatchNormalization, Dropout,
    multiply)
from keras.optimizers import Adam
from keras.utils import plot_model
from .conf import *

# 优化器
adam = Adam(lr, beta_1)

# 【真伪】标签
zero, one = 0, .95
label = np.array([one] * batch_size + [zero] * batch_size)


def load_data():
    (x, y), _ = mnist.load_data()
    x = x[:n_samples].reshape(-1, *shape) / 127.5 - 1
    return x, y[:n_samples].reshape(-1, 1)


class GAN:
    def __init__(self):
        self.generator = None  # 【生成器】
        self.judge = None  # 【审判者】
        self.tricker = None  # 【欺诈者】

    def modeling(self):
        self.build_generator()
        self.build_judge()
        self.build_tricker()

    def build_generator(self):
        """输入【噪音】和【类别】，输出【赝品】图像"""
        # 反卷积神经网络
        cnnt = Sequential()
        cnnt.add(Dense(3 * 3 * 384, input_dim=noise_dim, activation='relu'))
        cnnt.add(Reshape((3, 3, 384)))
        cnnt.add(Conv2DTranspose(192, 5, strides=1, padding='valid', activation='relu',
                                 kernel_initializer='glorot_normal'))
        cnnt.add(BatchNormalization())
        cnnt.add(Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu',
                                 kernel_initializer='glorot_normal'))
        cnnt.add(BatchNormalization())
        cnnt.add(Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh',
                                 kernel_initializer='glorot_normal'))

        # 【噪音】和【类别】
        noise = Input(shape=(noise_dim,))
        num = Input(shape=(1,), dtype='int32')
        num_emb = Embedding(num_classes, noise_dim,
                            embeddings_initializer='glorot_normal')(num)
        h = multiply([noise, num_emb])  # 哈达马积

        # 输出【赝品】图像
        x_fake = cnnt(h)
        self.generator = Model([noise, num], x_fake)

        # 模型可视化
        plot_model(cnnt, path_cnnt_imgs,
                   show_shapes=True, show_layer_names=False)
        plot_model(self.generator, path_generator_imgs,
                   show_shapes=True, show_layer_names=False)

    def build_judge(self):
        """输入图像，输出【真伪】和【类别】"""
        # 卷积神经网络
        cnn = Sequential()

        cnn.add(Conv2D(32, 3, padding='same', strides=2, input_shape=shape))
        cnn.add(LeakyReLU(.2))
        cnn.add(Dropout(.3))

        cnn.add(Conv2D(64, 3, padding='same', strides=1))
        cnn.add(LeakyReLU(.2))
        cnn.add(Dropout(.3))

        cnn.add(Conv2D(128, 3, padding='same', strides=2))
        cnn.add(LeakyReLU(.2))
        cnn.add(Dropout(.3))

        cnn.add(Conv2D(256, 3, padding='same', strides=1))
        cnn.add(LeakyReLU(.2))
        cnn.add(Dropout(.3))

        cnn.add(Flatten())

        # 输入图像，获取图像特征
        image = Input(shape)
        flatten = cnn(image)

        # 输出【真伪】和【类别】
        judgement = Dense(1, activation='sigmoid')(flatten)
        num = Dense(num_classes, activation='softmax')(flatten)

        # 编译
        self.judge = Model(image, [judgement, num])
        self.judge.compile(
            adam, ['binary_crossentropy', 'sparse_categorical_crossentropy'])

        # 模型可视化
        plot_model(cnn, path_cnn_imgs,
                   show_shapes=True, show_layer_names=False)
        plot_model(self.judge, path_judge_imgs,
                   show_shapes=True, show_layer_names=False)

    def build_tricker(self):
        """伪造【赝品】交给【审判者】，返回审判结果，据此提升【伪造技术】"""
        # 生成【赝品】图像
        noise = Input(shape=(noise_dim,))
        num_noise = Input(shape=(1,), dtype='int32')
        x_fake = self.generator([noise, num_noise])

        # 【审判者】不训练
        self.judge.trainable = False
        judgement, num_judgement = self.judge(x_fake)
        self.tricker = Model([noise, num_noise], [judgement, num_judgement])

        # 编译
        self.tricker.compile(
            adam, ['binary_crossentropy', 'sparse_categorical_crossentropy'])

        # 模型可视化
        plot_model(self.tricker, path_tricker_imgs,
                   show_shapes=True, show_layer_names=False)

    def train_judge(self, x, y):
        # 【赝品】图像制造
        noise = np.random.uniform(-1, 1, (batch_size, noise_dim))  # 均匀分布
        num_noise = np.random.randint(0, num_classes, (batch_size, 1))
        x_fake = self.generator.predict([noise, num_noise], verbose=0)

        # 真假图像合并
        x = np.concatenate((x, x_fake))
        num = np.concatenate((y, num_noise), axis=0)

        # 【类别】权重分配：【真实类别】权重为2，【噪音类别】权重为0，总体权重均值为1
        weight = [np.ones(2 * batch_size),
                  np.concatenate((np.ones(batch_size) * 2, np.zeros(batch_size)))]

        # 批训练，返回损失
        return self.judge.train_on_batch(x, [label, num], sample_weight=weight)

    def train_tricker(self):
        noise = np.random.uniform(-1, 1, (2 * batch_size, noise_dim))
        trick = np.ones(2 * batch_size) * 1
        num = np.random.randint(0, num_classes, (2 * batch_size, 1))
        return self.tricker.train_on_batch([noise, num], [trick, num])

    def train(self, x, y):
        loss_d = self.train_judge(x, y)  # 训练【审判者】
        loss_t = self.train_tricker()  # 训练【欺诈者】
        return loss_d, loss_t

    def save_fig(self, epoch):
        nrows, ncols = 10, 10
        noises = np.random.normal(size=(nrows * ncols, noise_dim))
        nums = np.array(list(range(nrows)) * ncols)
        imgs = self.generator.predict([noises, nums])
        imgs = .5 * imgs + .5  # 预处理还原
        for i in range(nrows):
            for j in range(ncols):
                mp.subplot(nrows, ncols, i * ncols + j + 1)
                mp.imshow(imgs[i * ncols + j].reshape(28, 28), cmap='gray')
                mp.axis('off')
        mp.savefig(path_imgs % epoch)
        mp.close()


if __name__ == '__main__':
    gan = GAN()
    gan.modeling()
    x, y = load_data()
    for e in range(epochs):
        print('\033[033m{} %s\033[0m'.format(e) %
              '[loss_d_total loss_d_2 loss_d_10] [loss_t_total loss_t_2 loss_t_10]')
        for i in range(times):
            loss_d, loss_t = gan.train(x[i * batch_size: (i + 1) * batch_size],
                                       y[i * batch_size: (i + 1) * batch_size])
            print(i, loss_d, loss_t)
        gan.save_fig(e)