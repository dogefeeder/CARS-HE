import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
from glob import glob
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from functools import partial
from threading import Thread
from tornado import gen
#
from keras.models import Model,Sequential
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

if os.path.exists('./images'):
    shutil.rmtree('./images')
os.mkdir('./images')
#
d_losses = []
g_losses = []
s_losses = []
accs_seg_losses = []
accs = []
# image shape
ims = 320
# Number of filters in the first layer of G and D
gf = 64
df = 64

# dices
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)





######### from data_loader import DataLoader ########
def one_hot(im,n_cl=3):
    im_sh = list(im.shape) + [n_cl,]
    im0 = np.zeros(im_sh,dtype=int)
    for i in range(n_cl):
        im0[:,:,i] = np.array(im==i,dtype=int)
    return im0
    
class DataLoader():
    def __init__(self):
        self.origs = sorted(glob('./augs256/orig/*.npy'))
        self.masks = sorted(glob('./augs256/mask/*.npy'))
        self.segs = sorted(glob('./augs256/seg/*.npy'))

    def load_batch(self, batch_size=1):
        imgs = []
        mks = []
        sgs = []
#         ind = np.random.choice(len(self.origs),batch_size,replace=False)
        i=0
        while i<batch_size:
            ind = np.random.choice(len(self.origs),1)[0]
            # imgs
            im = np.expand_dims(np.load(self.origs[ind]),-1)*2.-1
            if im.shape[0]!=im.shape[1]:
                continue
            imgs += [im,]
            # mks
            mk = np.load(self.masks[ind])
            mk[mk>150]=150
            mks += [mk/75.-1,]
            # sgs
            sg = one_hot(np.load(self.segs[ind]),3)
            sgs += [sg,]
            i+=1
        imgs = np.array(imgs) + np.random.uniform(-0.005,0.005,np.shape(imgs))
        imgs[imgs<-1]=-1
        imgs[imgs>1]=1
        mks = np.array(mks)# + np.random.uniform(-0.0005,0.0005,np.shape(segs))
        sgs = np.array(sgs)
#         imgs[imgs>1]=1.
#         segs[segs>1]=1.
        return mks,imgs,sgs

######## bokeh: real time visualization ###########
source = ColumnDataSource(data={'epochs': [],'d_loss': [],'g_loss': [],'s_loss': []})
plot = figure()
plot.line(x= 'epochs', y='d_loss',color='blue',alpha=0.8,source=source)
plot.line(x= 'epochs', y='g_loss',color='red', alpha=0.8,source=source)
plot.line(x= 'epochs', y='s_loss',color='orange', alpha=0.8,source=source)
doc = curdoc()
doc.add_root(plot)
@gen.coroutine
def update(new_data):
    source.stream(new_data)


######## Losses and accuracy #######
def LA_vis(h=10,w=6):
    # losses
    plt.subplots(4,1,figsize=(24,6))
    plt.subplot(411)
    plt.plot(np.array(d_losses),'bo',markersize=3,label = 'd_loss')
    plt.title('d_loss')
    plt.xlabel('i',size=15)
    plt.ylabel('Loss',size=15)
    plt.legend(fontsize=15)
    plt.subplot(412)
    plt.plot(np.array(g_losses),'ro',markersize=3,label = 'g_loss')
    plt.title('g_loss')
    plt.xlabel('i',size=15)
    plt.ylabel('Loss',size=15)
    plt.legend(fontsize=15)
    plt.subplot(413)
    plt.plot(np.array(s_losses),'ro',markersize=3,label = 's_loss')
    plt.title('s_loss')
    plt.xlabel('i',size=15)
    plt.ylabel('Loss',size=15)
    plt.legend(fontsize=15)
    plt.subplot(414)
    plt.plot(np.array(accs_seg_losses),'ro',markersize=3,label = 'acc_seg')
    plt.title('s_loss')
    plt.xlabel('i',size=15)
    plt.ylabel('Accuray',size=15)
    plt.legend(fontsize=15)
    plt.savefig('./losses.png')
    plt.close()
    # accs
    plt.figure(figsize=(10,6))
    plt.plot(np.array(accs)*100,'bo',markersize=3)
    plt.xlabel('i',size=15)
    plt.ylabel('Accuracy %',size=15)
    plt.savefig('./accuracy.png')
    plt.close()
    pd.DataFrame({'g_loss':g_losses,'s_loss':s_losses}).to_csv('./gs_loss_lr75e-5_bs4.csv')
    return 

########## Pix2Pix gan ##########
class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = ims
        self.img_cols = ims
        # seg
        self.img_shapeA = (self.img_rows, self.img_cols, 3)
        # orig
        self.img_shapeB = (self.img_rows, self.img_cols, 1)

        # Configure data loader
        self.data_loader = DataLoader()

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = gf
        self.df = df

        optimizer = Adam(0.00007, 0.6)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator,self.segmentor = self.build_generator()
        self.segmentor.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shapeA) # histo
        img_B = Input(shape=self.img_shapeB) # cars

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.75)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shapeB)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        
        #m
        m12 = deconv2d(d2, d1, self.gf)
        m23 = deconv2d(d3, d2, self.gf*2)
        m34 = deconv2d(d4, d3, self.gf*4)
        #mm
        mm16 = deconv2d(m23, m12, self.gf)
        #m
        m45 = deconv2d(m34, m23, self.gf*2)
        m56 = deconv2d(m45, mm16, self.gf)

        # Upsampling
        u3 = deconv2d(d5, d4, self.gf*8)
        u4 = deconv2d(u3, m34, self.gf*4)
        u5 = deconv2d(u4, m45, self.gf*2)
        u6 = deconv2d(u5, m56, self.gf)
        
        # m12
        m12_up = UpSampling2D(size=2)(m12)
        # mm16
        mm16_up = UpSampling2D(size=2)(mm16)
        # m56
        m56_up = UpSampling2D(size=2)(m56)
        # 
        u7 = UpSampling2D(size=2)(u6)
        # concatenate
        conca = Concatenate()([u7,m12_up,mm16_up,m56_up])
        output_img = Conv2D(self.img_shapeA[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(conca)
        
        #segmentation
        mm16_ = UpSampling2D(size=2)(mm16)
        output_seg = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='softmax')(mm16_)
        
        return Model(d0, output_img),Model(d0, output_seg)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shapeA)
        img_B = Input(shape=self.img_shapeB)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)
    
    def train(self, epochs=25000, batch_size=6, sample_interval=100):
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            imgs_A, imgs_B, imgs_seg = self.data_loader.load_batch(batch_size)
#             print(np.sum(imgs_A==np.nan),np.sum(imgs_B==np.nan))
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
            
            # -----------------
            #  Train Segmentation
            # -----------------

            # 
            seg_loss = self.segmentor.train_on_batch(imgs_B, imgs_seg)
            
            # save losses and accs
            d_losses.append(d_loss[0])
            accs.append(d_loss[1])
            g_losses.append(g_loss[0])
            s_losses.append(seg_loss[0])
            accs_seg_losses.append(seg_loss[1])
            # Plot the progress
#             print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f]" % (
#                                                         epoch, epochs,
#                                                         d_loss[0], 100*d_loss[1],
#                                                         g_loss[0]))
            # real time visualization
            new_data = {'epochs': [epoch],'d_loss': [d_loss[0]],'g_loss': [g_loss[0]*0.02],'s_loss': [seg_loss[0]/8.]}
            doc.add_next_tick_callback(partial(update, new_data))
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.generator.save('./Gen_cars.h5')
                self.discriminator.save('./Disc_cars.h5')
                self.segmentor.save('./seg_cars.h5')
        # save losses and accs histories
        LA_vis(h=10,w=6)


    def sample_images(self, epoch):
        r, c = 5, 3

        imgs_A, imgs_B, imgs_seg = self.data_loader.load_batch(batch_size=3)
        fake_A = self.generator.predict(imgs_B)
        seg_A = np.expand_dims(np.argmax(self.segmentor.predict(imgs_B),-1)-1,-1)
        imgs_seg = np.expand_dims(np.argmax(imgs_seg,-1)-1,-1)
        imgs_B = np.concatenate([-np.ones(imgs_B.shape),imgs_B,-np.ones(imgs_B.shape)],-1)
        seg_A = np.concatenate([seg_A,]*3,-1)
        imgs_seg = np.concatenate([imgs_seg,]*3,-1)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A, seg_A, imgs_seg])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Orig', 'Pseudo', 'Label', 'Seg_pred', 'Seg_label']
        fig, axs = plt.subplots(c,r,figsize=(r*3,c*3))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[j,i].imshow(gen_imgs[cnt])
                if j==0:
                    axs[j,i].set_title(titles[i],fontsize=15)
                axs[j,i].axis('off')
                cnt += 1
        fig.savefig("images/sample_%d.png" % (epoch))
        plt.close()


######## main function ##########
gan = Pix2Pix()
# training
# gan.train(epochs=1000, batch_size=4, sample_interval=20)
thread = Thread(target=gan.train)
thread.start()

# bokeh serve --show 3TrainingPix2Pix.py --allow-websocket-origin=34.220.246.0:5006 (or *)












