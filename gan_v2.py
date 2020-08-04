# coding = utf-8
# styleGAN

from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial, update_wrapper
from random import random

# 常量

img_size = 256
latent_size = 512
BATCH_SIZE = 4
n_img = 10000
img_path = './resized_img'
suff = 'jpg'

# 样式 Z向量


def noise(n):
    return np.random.normal(0.0, 0.1, size=[n, latent_size])


# 噪声样式


def noiseImage(n):
    return np.random.normal(0.0, 0.1, size=[n, img_size, img_size, 1])


# 随机获取矩阵


def get_rand(arr, n):
    idx = np.random.randint(0, arr.shape[0], n)
    return arr[idx]


# 导入图片


def import_images(path, flip=True, suffix='png'):
    out = []
    cont = True
    i = 1
    while (cont):
        try:
            temp = Image.open(path + '/' + str(i) + '.' + suffix)
            temp = temp.resize((img_size, img_size), Image.BICUBIC)
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            out.append(temp1)
            if flip:
                out.append(np.flip(out[-1], 1))
            i += 1
        except:
            cont = False
    print('Imported {0} images'.format(i - 1))
    return np.array(out)


# 归一化


def normalize(mat):
    # return (mat-np.mean(mat))/np.std(mat)
    return (mat - np.mean(mat)) / (np.std(mat) + 1e-7)


class ImageGenerator(object):
    def __init__(self, path, n, flip=True, suffix='png'):
        self.path = path
        self.n = n
        self.flip = flip
        self.suffix = suffix

    def get_batch(self, amount):
        idx = np.random.randint(0, self.n - 1, amount) + 1
        out = []
        for i in idx:
            temp = Image.open(img_path + '/' + str(i) + '.' + suff)
            temp = temp.resize((img_size, img_size), Image.BICUBIC)
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            # out.append(temp1)
            if self.flip and random() > 0.5:
                temp1 = np.flip(temp1, 1)
            out.append(temp1)
        return np.array(out)


from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
import keras.backend as K

from AdaIN import AdaInstanceNormalization

# 损失函数

def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                             axis=np.arange(1, len(gradients_sqr.shape)))
    return K.mean(gradient_penalty * weight)

# 生成器模块

def g_block(inp,style,noise,fil,u=True):

    # 1
    b=Dense(fil)(style)
    b=Reshape([1,1,fil])(b)
    g=Dense(fil)(style)
    g=Reshape([1,1,fil])(g)
    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    else:
        out = Activation('linear')(inp)
    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)

    # 2
    b=Dense(fil)(style)
    b=Reshape([1,1,fil])(b)
    g=Dense(fil)(style)
    g=Reshape([1,1,fil])(g)
    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)
    return out

# 判别器模块

def d_block(inp, fil, p = True):

    # 1
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    route2 = LeakyReLU(0.01)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    
    # 2
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
    out = LeakyReLU(0.01)(route2)
    return out

# GAN

class GAN(object):
    
    def __init__(self,lr=0.001):
        self.D=None
        self.G=None
        self.DM=None
        self.GM=None
        self.LR=lr
        self.steps=1
        self.discriminator()
        self.generator()
    
    def discriminator(self):
        if self.D:
            return self.D
        inp=Input(shape=[img_size,img_size,3])
        x=d_block(inp,16)
        x=d_block(x,32)
        x=d_block(x,64)
        if (img_size > 32):
            x = d_block(x, 128) 
        if (img_size > 64):
            x = d_block(x, 192) 
        if (img_size > 128):
            x = d_block(x, 256) 
        if (img_size > 256):
            x = d_block(x, 384) 
        if (img_size > 512):
            x = d_block(x, 512) 
        x=Flatten()(x)
        x=Dense(128)(x)
        x=Activation('relu')(x)
        x=Dropout(0.6)(x)
        x=Dense(1)(x)
        self.D=Model(inputs=inp,outputs=x)
        return self.D

    def generator(self):
        if self.G:
            return self.G

        # 生成 W

        inp_s=Input(shape=[latent_size])
        sty = Dense(512, kernel_initializer = 'he_normal')(inp_s)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)

        # 噪声

        inp_n = Input(shape = [img_size, img_size, 1])
        noi = [Activation('linear')(inp_n)]
        curr_size = img_size
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size/2))(noi[-1]))

        # 主体

        inp = Input(shape = [1])
        x = Dense(4 * 4 * 512, kernel_initializer = 'he_normal')(inp)
        x = Reshape([4, 4, 512])(x)
        x = g_block(x, sty, noi[-1], 512, u=False)

        if(img_size >= 1024):
            x = g_block(x, sty, noi[7], 512) # Size / 64
        if(img_size >= 512):
            x = g_block(x, sty, noi[6], 384) # Size / 64
        if(img_size >= 256):
            x = g_block(x, sty, noi[5], 256) # Size / 32
        if(img_size >= 128):
            x = g_block(x, sty, noi[4], 192) # Size / 16
        if(img_size >= 64):
            x = g_block(x, sty, noi[3], 128) # Size / 8
            
        x = g_block(x, sty, noi[2], 64) # Size / 4
        x = g_block(x, sty, noi[1], 32) # Size / 2
        x = g_block(x, sty, noi[0], 16) # Size
        
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)
        
        self.G = Model(inputs = [inp_s, inp_n, inp], outputs = x)
        
        return self.G

    def AdModel(self):
        
        self.D.trainable=False
        for layer in self.D.layers:
            layer.trainable=False
        
        self.G.trainable=True
        for layer in self.G.layers:
            layer.trainable=True

        inp1=Input(shape=[latent_size])
        inp2=Input(shape=[img_size,img_size,1])
        inp3=Input(shape=[1])

        g_out=self.G([inp1,inp2,inp3])
        d_out=self.D(g_out)

        self.AM=Model(inputs=[inp1,inp2,inp3],outputs=g_out)
        self.AM.compile(optimizer = Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')

        return self.AM

    def DisModel(self):

        self.D.trainable=True
        for layer in self.D.layers:
            layer.trainable=True
        
        self.G.trainable=False
        for layer in self.G.layers:
            layer.trainable=False
        
        # 真实数据

        real_img=Input(shape=[img_size,img_size,3])
        real_out=self.D(real_img)

        # 生成数据

        inp1=Input(shape=[latent_size])
        inp2=Input(shape=[img_size,img_size,1])
        inp3=Input(shape=[1])

        fake_img=self.G([inp1,inp2,inp3])
        fake_out=self.D(fake_img)

        all_out=self.D(real_img)

        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = real_img, weight = 5)
        partial_gp_loss = update_wrapper(partial_gp_loss, gradient_penalty_loss)

        self.DM=Model(inputs=[real_img,inp1,inp2,inp3],outputs=[real_out,fake_out,all_out])
        self.DM.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])

        return self.DM

class WGAN(object):

    def __init__(self,steps=-1,lr=0.0001,silent=True):
        self.GAN=GAN(lr=lr)
        self.DisModel=self.GAN.DisModel()
        self.AdModel=self.GAN.AdModel()
        self.generator=self.GAN.generator()

        if steps>0:
            self.GAN.steps=steps

        self.lastblip=time.perf_counter()
        self.noise_level=0

        self.im=ImageGenerator(img_path,n_img,suffix=suff,flip=True)

        self.silent=silent

        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

        self.enoise = noise(8)
        self.enoiseImage = noiseImage(8)

    def train(self):

        a=self.train_dis()
        b=self.train_gen()

        # 输出损失
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(b))
            s = round((time.perf_counter() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.perf_counter()
            
            # 保存模型
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            # 生成图片
            if self.GAN.steps % 1000 == 0:
                self.evaluate(floor(self.GAN.steps / 1000))
            
        self.GAN.steps = self.GAN.steps + 1

    def train_dis(self):
        train_data = [self.im.get_batch(BATCH_SIZE), noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones]
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])
        return d_loss

    def train_gen(self):
        g_loss = self.AdModel.train_on_batch([noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones], self.zeros)
        return g_loss

    def evaluate(self,num=0,trunc=2.0):
        n=noise(32)
        n2=noiseImage(32)
        im2 = self.generator.predict([n, n2, np.ones([32, 1])])
        im3 = self.generator.predict([self.enoise, self.enoiseImage, np.ones([8, 1])])
        
        r12 = np.concatenate(im2[:8], axis = 1)
        r22 = np.concatenate(im2[8:16], axis = 1)
        r32 = np.concatenate(im2[16:24], axis = 1)
        r43 = np.concatenate(im3[:8], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))
        
        x.save('./gen_img/'+str(num)+'.jpg')

    def evalTrunc(self, num = 0, trunc = 1.8):
        
        n = np.clip(noise(16), -trunc, trunc)
        n2 = noiseImage(16)
        
        im2 = self.generator.predict([n, n2, np.ones([16, 1])])
        
        r12 = np.concatenate(im2[:4], axis = 1)
        r22 = np.concatenate(im2[4:8], axis = 1)
        r32 = np.concatenate(im2[8:12], axis = 1)
        r43 = np.concatenate(im2[12:], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))
        
        x.save('./gen_img/'+str(num)+'.jpg')

    def saveModel(self, model, name, num): #Save a Model
        json = model.to_json()
        with open("model/"+name+".json", "w") as json_file:
            json_file.write(json)
        model.save_weights("model/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num): #Load a Model
        
        file = open("model/"+name+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json, custom_objects = {'AdaInstanceNormalization': AdaInstanceNormalization})
        mod.load_weights("model/"+name+"_"+str(num)+".h5")
        
        return mod        

    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Load Models
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)
        
        self.GAN.steps = steps1
        
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()

if __name__ == "__main__":
    model = WGAN(lr = 0.0003, silent = False)
    # model.load(219)
    
    # for i in range(10000):
    #     model.evalTrunc(i)
    
    # while(False):
    #    model.train()

    for i in range(1000):
        model.train()
