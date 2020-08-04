# -*- coding: utf-8 -*-

print('V3')

import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from math import log2

import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
import keras.backend as K

img_size = 32
latent_size = 32
batch_size = 32
iterations = 50000

assert log2(img_size).is_integer, 'input image size must be a power of 2'
n_layers = int(log2(img_size))

def noise(batch_size, latent_size):
    return np.random.normal(0.0, 1.0, size=[batch_size, latent_size]).astype(float)

def noise_list(batch_size, n_layers, latent_size):
    return [noise(batch_size, latent_size)] * n_layers

def random_weighted_average(imgs):
    alpha = K.random_uniform((32, 1, 1, 1))
    return (alpha * imgs[0]) + ((1 - alpha) * imgs[1])
    
# mixing regularization
def mixed_list(n, layers, latent_size):
    break_point = int(random() * layers)
    return noise_list(n, break_point, latent_size) + noise_list(n, layers - break_point, latent_size)

def gradient_penalty(real_img, fake_img, averaged_samples):
    gradients = K.gradients(fake_img, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradients_sqr_sum)

# Adaptive Instance Normalization
def AdaIN(img):
    mean = K.mean(img[0], axis=[0, 1], keepdims=True)
    std = K.std(img[0], axis=[0, 1], keepdims=True)
    out = (img[0] - mean) / std
    
    pool_shape = [-1, 1, 1, out.shape[-1]]
    scale = K.reshape(img[1], pool_shape)
    bias = K.reshape(img[2], pool_shape)
    
    return out * scale + bias

def g_block(inp_tensor, latent_vector, filters):
    scale = Dense(filters)(latent_vector)
    bias = Dense(filters)(latent_vector)
    
    out = UpSampling2D()(inp_tensor)
    out = Conv2D(filters, 3, padding='same')(out)
    out = Lambda(AdaIN)([out, scale, bias])
    out = LeakyReLU(alpha=0.2)(out)
    
    return out

def d_block(inp_tensor, filters):
    out = Conv2D(filters, 3, padding='same')(inp_tensor)
    out = LeakyReLU(alpha=0.2)(out)
    #out = Conv2D(filters, 3, padding='same')(out)
    #out = LeakyReLU(alpha=0.2)(out)
    out = AveragePooling2D()(out)
    
    return out

class StyleGAN():
    
    def __init__(self, steps=1, lr=0.0001, latent_size=latent_size, n_layers=n_layers, img_size=img_size):
        self.latent_size = latent_size
        self.steps = 1
        self.lr = lr
        self.n_layers = n_layers
        self.img_size = img_size
        optimizer = RMSprop(lr=0.00005)

        # build generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Discriminator Computational Graph
        
        # freeze generator layers while training discriminator
        self.generator.trainable = False
        
        # image input
        real_img = Input([self.img_size, self.img_size, 3], name='real_image')
        
        # latent vector
        z = Input([self.latent_size])
        # generate image using latent vector
        fake_img = self.generator(z)
        
        # discriminator determines validity
        fake = self.discriminator(fake_img)
        valid = self.discriminator(real_img)
        
        # weighted average between real and fake
        interpolated_img = random_weighted_average([real_img, fake_img])
        valid_interpolated = self.discriminator(interpolated_img)
        
        partial_gp_loss = partial(gradient_penalty, averaged_samples=real_img)
        partial_gp_loss.__name__ = 'gradient_penalty_loss'
        
        self.discriminator_model = Model(inputs=[real_img, z], outputs=[valid, fake, valid])
        self.discriminator_model.compile(optimizer=optimizer, loss=['mse', 'mse', partial_gp_loss], loss_weights=[1,1,10])
        
        # Generator Computational Graph
        self.discriminator.trainable = False
        self.generator.trainable = True
        
        # latent vector
        z_gen = Input([self.latent_size])
        # generate image based on vector
        gen_img = self.generator(z_gen)
        # discriminator determines validity
        valid = self.discriminator(gen_img)
        # define generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(optimizer=optimizer, loss='mse')
        
    def build_generator(self):
        latent_input = Input(shape=[self.latent_size])
        
        # latent mapping network
        latent = Dense(64)(latent_input)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Dense(64)(latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        latent = Dense(64)(latent)
        latent = LeakyReLU(alpha=0.2)(latent)
        
        out = Dense(4*4*32, activation='relu')(latent_input)
        out = Reshape([4, 4, 32])(out)
        
        # out = g_block(out, latent, 64)
        out = g_block(out, latent, 32) 
        out = g_block(out, latent, 16) 
        out = g_block(out, latent, 8)
        img_output = Conv2D(3, 1, padding='same', activation='sigmoid')(out)
        
        generator_model = Model(inputs=latent_input, outputs=img_output)
        #print("Generator Model")
        #generator_model.summary()
        
        return generator_model
    
    def build_discriminator(self):
        img_input = Input(shape=[self.img_size, self.img_size, 3])
        out = d_block(img_input, 16)
        out = d_block(img_input, 32) # ERROR
        out = d_block(out, 64)
        
        out = Flatten()(out)
        
        out = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros')(out)
        out = LeakyReLU(alpha=0.02)(out)
        out = Dropout(0.2)(out)
        out = Dense(1, kernel_initializer='he_normal', bias_initializer='zeros')(out)
        
        discriminator_model = Model(inputs=img_input, outputs=out)
        #print("Discriminator Model")
        #discriminator_model.summary()
        
        return discriminator_model
    
    def train(self, epochs, batch_size, sample_interval=100):
        (X_train, _), (_, _) = cifar10.load_data()
        
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        
        valid = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])
        # dummy labels for gradient penalty
        dummy = np.ones([batch_size, 1])
        
        for epoch in range(epochs):

            # train discriminator
            
            # random sample of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # generator input).
            z = noise(batch_size, self.latent_size)
            # train discriminator
            d_loss = self.discriminator_model.train_on_batch([imgs, z], [valid, fake, dummy])
            
            # train generator
            g_loss = self.generator_model.train_on_batch(z, valid)
                        
            if epoch % sample_interval == 0:
                print("{0} [Discriminator loss: {1}] [Generator loss: {2}]".format(epoch, d_loss[0], g_loss))
                self.sample_images(epoch)
    
    def sample_images(self, epoch):
        rows, cols = 5, 5
        noise = np.random.normal(0, 1, (rows * cols, self.latent_size))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for i in range(rows):
            for j in range(cols):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        #fig.savefig("images/cifar_%d.png" % epoch)
        plt.close()
        
if __name__ == '__main__':
    stylegan = StyleGAN()
    stylegan.generator.summary()
    stylegan.discriminator.summary()
    stylegan.train(epochs=50000, batch_size=batch_size)
        