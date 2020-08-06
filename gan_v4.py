# coding: utf-8

import os
import numpy as np
from functools import partial
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
from keras import preprocessing
from math import log2
from PIL import Image
import keras.backend as K
import matplotlib.pyplot as plt
import datetime

img_size = 32
latent_size = 32
batch_size = 20
iterations = 50000
n_layers = int(log2(img_size))


def load_data():
    file_list = os.listdir('./resized_img')
    out = []
    for file in file_list:
        img = Image.open('./resized_img/{0}'.format(file)).convert('RGB')
        data = np.reshape(np.array(img.getdata()), (img_size, img_size, 3))
        out.append(data)
    print('Load {0} Pics'.format(len(out)))
    return np.array(out)


class ImageFlow:
    def __init__(self):
        self.image_list = os.listdir('./resized_img/')
        self.amount = len(self.image_list)
        print('Found {} Pics'.format(self.amount))

    def get(self, batch_size):
        idx = np.random.randint(0, self.amount, batch_size)
        out = []
        for i in idx:
            img = Image.open(
                os.getcwd() +
                '/resized_img/{0}'.format(self.image_list[i])).convert('RGB')
            data = np.reshape(np.array(img.getdata()), (img_size, img_size, 3))
            out.append(data)
        return np.array(out)


def AdaIN(img):
    mean = K.mean(img[0], axis=[0, 1], keepdims=True)
    std = K.std(img[0], axis=[0, 1], keepdims=True)
    out = (img[0] - mean) / std

    pool_shape = [-1, 1, 1, out.shape[-1]]
    scale = K.reshape(img[1], pool_shape)
    bias = K.reshape(img[2], pool_shape)

    return (out * scale) + bias


def g_block(inp, latent, filters):

    scale = Dense(filters)(latent)
    bias = Dense(filters)(latent)

    out = UpSampling2D()(inp)
    out = Conv2D(filters, 3, padding='same')(out)
    out = Lambda(AdaIN)([out, scale, bias])
    out = LeakyReLU(alpha=0.2)(out)
    return out


def d_block(inp, filters):
    out = Conv2D(filters, 3, padding='same')(inp)
    out = LeakyReLU(alpha=0.3)(out)
    return out


def noise(batch_size, latent_size):
    return np.random.normal(0.0, 1.0, size=[batch_size,
                                            latent_size]).astype(float)


def gradient_penalty(real_img, fake_img, averaged_samples):
    gradients = K.gradients(fake_img, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradients_sqr_sum)


class styleGAN():
    def __init__(self,
                 lr=0.0001,
                 latent_size=latent_size,
                 img_size=img_size,
                 img_path='./resized_img',
                 log_path=None):
        self.lr = lr
        self.latent_size = latent_size
        self.img_size = img_size
        self.img_path = img_path
        self.optimizers = RMSprop(lr=0.00005)
        self.log = open(log_path, 'w')

        self.generator = self.build_generator()
        print('Generator : ')
        self.generator.summary()
        self.discriminator = self.build_discriminator()
        print('Discriminator : ')
        self.discriminator.summary()

        self.generator.trainable = False
        real_img = Input(shape=[self.img_size, self.img_size, 3],
                         name='real_img')
        z = Input(shape=[self.latent_size])
        fake_img = self.generator(z)
        real_out = self.discriminator(real_img)
        fake_out = self.discriminator(fake_img)

        partial_gp_loss = partial(gradient_penalty, averaged_samples=real_img)
        partial_gp_loss.__name__ = 'gradient_penalty_loss'  # partial后没有__name__属性

        self.d_train_model = Model(inputs=[real_img, z],
                                   outputs=[real_out, fake_out, real_out])
        self.d_train_model.compile(optimizer=self.optimizers,
                                   loss=['mse', 'mse', 'binary_crossentropy'],
                                   loss_weights=[1, 1, 10])
        self.generator.trainable = True
        self.discriminator.trainable = False

        z_gen = Input(shape=[self.latent_size])
        gen_img = self.generator(z_gen)
        o = self.discriminator(gen_img)
        self.g_train_model = Model(z_gen, o)
        self.g_train_model.compile(optimizer=self.optimizers, loss='mse')

    def train(self, epochs, batch_size, sample_interval=100):

        time_start = datetime.datetime.now()
        print('训练开始时间 : {0}'.format(time_start))
        self.log.write('Start at : {0}'.format(time_start))

        # x_train = load_data()
        # x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        x_train = ImageFlow()

        real_out = np.ones([batch_size, 1])
        fake_out = np.zeros([batch_size, 1])
        gp_out = np.ones([batch_size, 1])

        for epoch in range(epochs):
            # idx = np.random.randint(0, x_train.shape[0], batch_size)
            # data=x_train[idx]
            data = (x_train.get(batch_size=batch_size) - 127.5) / 127.5
            z = noise(batch_size=batch_size, latent_size=latent_size)
            d_loss = self.d_train_model.train_on_batch(
                [data, z], [real_out, fake_out, gp_out])
            g_loss = self.g_train_model.train_on_batch(z, real_out)
            if epoch % sample_interval == 0:
                print(
                    'Epoch {0}\nDiscriminator Loss : {1:.4f}\nGenerator Loss : {2:.4f}'
                    .format(epoch, d_loss[0], g_loss))
                self.log.write(
                    'Epoch {0}\nDiscriminator Loss : {1:.4f}\nGenerator Loss : {2:.4f}\n'
                    .format(epoch, d_loss[0], g_loss))
                self.log.flush()
                sample_img = self.generator.predict(
                    noise(batch_size=1, latent_size=latent_size))
                self.gen_sample_image(epoch)
                self.discriminator.save('./model/discriminator.h5')
                self.generator.save('./model/generator.h5')

        time_end = datetime.datetime.now()
        print('训练结束时间 : {0}'.format(time_end))
        print('训练耗时 : {0}'.format(time_end - time_start))
        self.log.write('End at : {0}\nTime cost : {1}\n'.format(time_end -
                                                              time_start))

    def build_generator(self):
        latent_input = Input(shape=[self.latent_size])
        latent = Dense(64)(latent_input)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Dense(64)(latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Dense(64)(latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        out = Dense(4 * 4 * 32, activation='relu')(latent_input)
        out = Reshape([4, 4, 32])(out)
        out = g_block(out, latent, 32)
        out = g_block(out, latent, 16)
        out = g_block(out, latent, 8)
        img_output = Conv2D(3, 1, padding='same', activation='sigmoid')(out)
        g_model = Model(inputs=latent_input, outputs=img_output)
        return g_model

    def build_discriminator(self):
        img_input = Input(shape=[self.img_size, self.img_size, 3])
        out = d_block(img_input, 16)
        out = d_block(out, 32)
        out = d_block(img_input, 64)
        out = Flatten()(out)
        out = Dense(128,
                    kernel_initializer='he_normal',
                    bias_initializer='zeros')(out)
        out = LeakyReLU(alpha=0.02)(out)
        out = Dropout(0.2)(out)
        out = Dense(1,
                    kernel_initializer='he_normal',
                    bias_initializer='zeros')(out)
        d_model = Model(inputs=img_input, outputs=out)
        return d_model

    def gen_sample_image(self, epoch):
        rows, cols = 4, 4
        noise = np.random.normal(0, 1, (rows * cols, self.latent_size))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5  # 复原像素值
        r1 = np.concatenate(gen_imgs[:4], axis=1)
        r2 = np.concatenate(gen_imgs[4:8], axis=1)
        r3 = np.concatenate(gen_imgs[8:12], axis=1)
        r4 = np.concatenate(gen_imgs[12:16], axis=1)
        all_mat = np.concatenate([r1, r2, r3, r4], axis=0)
        image = preprocessing.image.array_to_img(all_mat)
        image.save('./gen_img/{}.jpg'.format(epoch))


if __name__ == '__main__':
    stylegan = styleGAN(log_path='./log.txt')
    stylegan.train(epochs=25000, batch_size=batch_size, sample_interval=100)
