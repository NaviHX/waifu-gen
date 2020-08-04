# coding = utf-8
import keras
from keras.models import Model
from keras import layers
import numpy as np
import os
import random
from PIL import Image

noise_dim = 16  # 噪声维度
height = 32
width = 32
channels = 3

# G net

generator_input = keras.Input(shape=(noise_dim, ))

out = layers.Dense(16 * 16 * 128)(generator_input)
out = layers.LeakyReLU()(out)
out = layers.Reshape((16, 16, 128))(out)

out = layers.Conv2D(256, 5, padding='same')(out)
out = layers.LeakyReLU()(out)

out = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(out)
out = layers.LeakyReLU()(out)

out = layers.Conv2D(256, 5, padding='same')(out)
out = layers.LeakyReLU()(out)

out = layers.Conv2D(channels, 7, activation='tanh', padding='same')(out)
g_net = Model(generator_input, out)

# D net

discriminator_input = keras.Input(shape=(height, width, channels))

out = layers.Conv2D(32, 3)(discriminator_input)
out = layers.LeakyReLU()(out)

out = layers.Conv2D(32, 4, strides=2)(out)
out = layers.LeakyReLU()(out)

out = layers.Conv2D(32, 4, strides=2)(out)
out = layers.LeakyReLU()(out)

out = layers.Conv2D(32, 4, strides=2)(out)
out = layers.LeakyReLU()(out)

out = layers.Flatten()(out)
out = layers.Dropout(0.4)(out)

out = layers.Dense(1, activation='sigmoid')(out)

d_net = Model(discriminator_input, out)

# 网络构建

# 添加优化器

#discriminator_optimizer = keras.optimizers.RMSprop(learning_rate=0.8,
#                                                   clipvalue=1.0,
#                                                   decay=1e-8)
discriminator_optimizer = keras.optimizers.adam(learning_rate=0.01)
d_net.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# 链接生成器和判别器

# 固定判别器

d_net.trainable = False

gan_input = keras.Input(shape=(noise_dim, ))
gan_output = d_net(g_net(gan_input))
gan = Model(inputs=gan_input, outputs=gan_output)

#gan_optimizer = keras.optimizers.RMSprop(learning_rate=0.8,
#                                         clipvalue=1.0,
#                                         decay=1e-8)
gan_optimizer = keras.optimizers.adam(learning_rate=0.01)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# 训练准备

batch_size = 20
epoch = 500
img_path = 'D:\\GAN\\resized_img\\'
d_training_iter = 5  # 每一次迭代训练判别器的次数
log = open('log.txt', 'w')

img_list = os.listdir(img_path)
random.shuffle(img_list)
print('Found {0} images in {1}'.format(len(img_list), img_path))
start = 0

# 训练

for step in range(epoch):
    print('Step {0}'.format(step))
    stop = start + batch_size

    # 通过随机噪声生成假图片
    noise_vector = np.random.normal(size=(batch_size, noise_dim))
    fake_pic = g_net.predict(noise_vector)

    # 选取真实图片
    chosen = img_list[start:stop]
    true_pic = np.ndarray((batch_size, width, height, 3),dtype='float')
    for i in range(batch_size):
        true_pic[i] = np.array(Image.open(img_path + chosen[i]).convert('RGB'),dtype='float')

    pics = np.concatenate([fake_pic, true_pic])
    labels = np.concatenate(
        [np.ones(shape=(batch_size, 1)),
         np.zeros(shape=(batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)

    for i in range(d_training_iter):
        d_loss = d_net.train_on_batch(pics, labels)

    noise_vector = np.random.normal(size=(batch_size, noise_dim))
    noise_labels = np.zeros(shape=(batch_size, 1))

    a_loss = gan.train_on_batch(noise_vector, noise_labels)

    start += batch_size
    if start > len(img_list) - batch_size:
        start = 0

    if step % 10 == 0:  
        file = keras.preprocessing.image.array_to_img(fake_pic[0])
        file.save('D:/GAN/gen_img/{0}.png'.format(step))
        print('Discriminator loss : {0:.4f}'.format(d_loss))
        print('Generator loss : {0:.4f}'.format(a_loss))
        log.write(
            'Step : {0}\nDiscriminator loss : {1:.4f}\nGenerator loss : {2:.4f}\n'
            .format(step, d_loss, a_loss))

    if step % 100 == 0:
        gan.save('D:/GAN/model/gan.h5')

gan.save('D:/GAN/model/gan.h5')
g_net.save('D:/GAN/model/generator.h5')
print('END')
print('GAN Model saved at "./model/gan.h5"')
