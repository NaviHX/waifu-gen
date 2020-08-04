import keras
from keras import layers
import numpy as np
 
latent_dim = 32
height = 32
width = 32
channels = 3
 
generator_input = keras.Input(shape=(latent_dim,))
 
# 首先，将输入转换为16x16 128通道的feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)
 
# 然后，添加卷积层
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
 
# 上采样至 32 x 32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
 
# 添加更多的卷积层
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
 
# 生成一个 32x32 1-channel 的feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
 
# 重要的技巧（添加一个dropout层）
x = layers.Dropout(0,4)(x)
 
# 分类层
x = layers.Dense(1, activation='sigmoid')(x)
 
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()
 
discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# 将鉴别器（discrimitor）权重设置为不可训练（仅适用于`gan`模型）
discriminator.trainable = False
 
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
 
gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

import os
from keras.preprocessing import image
 
# 导入CIFAR10数据集
# (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

# 导入数据
import img_src
x_train=img_src.get_training_data()

# 从CIFAR10数据集中选择frog类（class 6）
# x_train = x_train[y_train.flatten() == 6]

# 标准化数据
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 20000
batch_size = 20
save_dir = '.\\gen_img'
 
start = 0 
# 开始训练迭代
for step in range(iterations):
    # 在潜在空间中抽样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    # 将随机抽样点解码为假图像
    generated_images = generator.predict(random_latent_vectors)
    
    # 将假图像与真实图像进行比较
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    # 组装区别真假图像的标签
    labels = np.concatenate([np.ones((batch_size, 1)),
                            np.zeros((batch_size, 1))])
    # 重要的技巧，在标签上添加随机噪声
    labels += 0.05 * np.random.random(labels.shape)
    
    # 训练鉴别器（discrimitor）
    d_loss = discriminator.train_on_batch(combined_images, labels)
    
    # 在潜在空间中采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    
    # 汇集标有“所有真实图像”的标签
    misleading_targets = np.zeros((batch_size, 1))
    
    # 训练生成器（generator）（通过gan模型，鉴别器（discrimitor）权值被冻结）
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:
        # 保存网络权值
        gan.save('./model/gan.h5')
        generator.save('./model/generator.h5')
 
        # 输出metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))
 
        # 保存生成的图像
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_img' + str(step) + '.png'))
