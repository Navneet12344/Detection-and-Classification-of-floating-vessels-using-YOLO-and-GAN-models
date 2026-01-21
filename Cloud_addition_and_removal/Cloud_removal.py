from PIL import Image
import os
from tensorflow import keras
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from matplotlib import pyplot as plt
from keras.utils import plot_model
from os import listdir
from numpy import asarray, load
from numpy import vstack
from keras.utils import img_to_array
from keras.utils import load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np
from skimage.metrics import structural_similarity as ssim ,mean_squared_error as mse, peak_signal_noise_ratio as psnr
from keras import backend as K
from keras.models import load_model
from numpy import load, corrcoef
from numpy import vstack
from numpy.random import randint
from math import log10, sqrt, ceil
import pandas as pd
import tensorflow as tf
import shutil
import splitfolders
from keras.engine.data_adapter import data_utils
from datetime import datetime 
import xlsxwriter

#log all losses
d_loss1_log = []
d_loss2_log = []
g_loss_log = []
psnr_before_log = []
psnr_after_log = []
ssim_before_log = []
ssim_after_log = []
mse_before_log = []
mse_after_log = []
averaged_psnr_before_values = []
averaged_psnr_after_values = []
averaged_ssim_before_values = []
averaged_ssim_after_values = []
averaged_mse_before_values = []
averaged_mse_after_values = []
averaged_dloss1_values=[]
averaged_dloss2_values=[]
averaged_gloss_values=[]

#SSIM metric calculation
def ssim_metric(target, generated):
    ssim1 = tf.image.ssim(target, generated, max_val=1)
    return ssim1

#PSNR Metric Calculation 
def psnr_metric(target, generated):
    mse = np.mean((target - generated)**2)
    if (mse == 0):
    #This means there is no difference between the pixel values/DN values of the source
    #image and the target image. Hence 100% sound and 0% noise
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def mean_squared_error(target, generated):
    mse = tf.losses.mean_squared_error(target, generated)
    return K.mean(mse)

def define_discriminator(in_shape=(256,256,3)):
    init = RandomNormal(stddev=0.02)
    # image input
    src_image = Input(shape=in_shape) # 256 x 256 x 3
    target_image= Input(shape=in_shape) # 256 x 256 x 3
    print(src_image.shape)
    # concat label as a channel
    merge = Concatenate()([src_image, target_image]) # 256 x 256 x 6 (6 channels, 3 for image and the other for another image)
    print(merge.shape)  
    # downsample
    #C64
    fe = Conv2D(64, (4,4), strides=(2,2), padding='same')(merge) # 128 x 128 x 64
    fe = LeakyReLU(alpha=0.2)(fe)
    print(fe.shape)
    # downsample
    #C128
    fe = Conv2D(128, (4,4), strides=(2,2), padding='same')(fe) # 64 x 64 x 128
    fe=BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    print(fe.shape)
    #C256
    fe = Conv2D(256, (4,4), strides=(2,2), padding='same')(fe) # 32 x 32 x 256
    fe=BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    print(fe.shape)
    #C512
    fe = Conv2D(512, (4,4), strides=(2,2), padding='same')(fe) # 16 x 16 x 512
    fe=BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe= Dropout(0.4)(fe)
    print(fe.shape)
    # #C512
    # fe = Conv2D(512, (4,4), padding='same')(fe) # 16 x 16 x 512
    # fe=BatchNormalization()(fe)
    # fe = LeakyReLU(alpha=0.2)(fe)
    # fe= Dropout(0.4)(fe)
    # print(fe.shape)


    #output
    fe = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(fe)  # 16 x 16 x 1 
    out_layer = Activation('sigmoid')(fe)
    
    # define model
    ## Combine input label with input image and supply as inputs to the model. 
    model = Model([src_image, target_image], out_layer)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_encoder_block(layer_in, n_filters, batchnorm=True):
    '''
    This function protrays the architecture of an encoder block
    '''
    # weight initialization
    init = RandomNormal(stddev=0.02)

    #add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)  	# add downsampling layer

    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)

    #Activating Leaky RelU
    g = LeakyReLU(alpha=0.2)(g) 	
    return g

# DECODER BLOCK

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    '''
    This function portrays the architecture of a decoder block
    '''
    #weight initialization
    init = RandomNormal(stddev=0.02)

    #add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in) 

    #add batch normalization
    g = BatchNormalization()(g, training=True)  	

    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
        
    # merge with skip connection
    #basically we concatenate the layers produced by (upconvolution) and the original layer in encoder block
    g = Concatenate()([g, skip_in]) 

    # relu activation
    g = Activation('relu')(g)
    return g

def define_generator(image_shape=(256,256,3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)    #128*128*64
    e2 = define_encoder_block(e1, 128)   # 64 x 64 x 128
    e3 = define_encoder_block(e2, 256)   # 32 x 32 x 256
    e4 = define_encoder_block(e3, 512)   # 16 x 16 x 512
    e5 = define_encoder_block(e4, 512)   # 8 x 8 x 512
    e6 = define_encoder_block(e5, 512)   # 4 x 4 x 512
    e7 = define_encoder_block(e6, 512)   # 2 x 2 x 512
    
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)  # 1 x 1 x 1024
    b = Activation('relu')(b)
    
    # decoder model
    d1 = decoder_block(b, e7, 512)    # 2 x 2 x 1024
    d2 = decoder_block(d1, e6, 512)   # 4 x 4 x 1024
    d3 = decoder_block(d2, e5, 512)   # 8 x 8  x 1024
    d4 = decoder_block(d3, e4, 512, dropout=False)    # 16 x 16 x 1024
    d5 = decoder_block(d4, e3, 256, dropout=False)    # 32 x 32 x 512
    d6 = decoder_block(d5, e2, 128, dropout=False)    # 64 x 64 x 256
    d7 = decoder_block(d6, e1, 64, dropout=False)     # 128 x 128 x 128

    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), activation='tanh', padding='same')(d7) # 256 x 256 x 3
    out_layer=Activation('tanh')(g)
    # define model
    model = Model(in_image, out_layer)
    return model   #Model not compiled as it is not directly trained like the discriminator.

# #Generator is trained via GAN combined model. 
# define the combined generator and discriminator model, for updating the generator
#Discriminator is trained separately so here only generator will be trained by keeping
#the discriminator constant. 
def define_gan(g_model, d_model,image_shape):
	d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
		
	# define the source image
	in_src = Input(shape=image_shape)
	
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    '''
    This functions selects a batch of random samples and returns an image with its target
    args --> image_stack, number of samples, patch shape
    '''
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    '''
    This function uses the generator model to generate fake samples 
    args --> generator model, number of samples, patch shape
    '''
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def summarize_performance(step, g_model, d_model, gan_model, dataset, len_trainA, n_samples=3):
    epoch_no = (step + 1) / len_trainA 
    #if we load model then epoch_no will be started from the last completed epoch
    # epoch_no = last_epoch + (step + 1) / len_trainA 
     
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
      plt.subplot(3, n_samples, 1 + i)
      plt.axis('off')
      plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
      plt.subplot(3, n_samples, 1 + n_samples + i)
      plt.axis('off')
      plt.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
      plt.subplot(3, n_samples, 1 + n_samples*2 + i)
      plt.axis('off')
      plt.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_best.png'
    # filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = 'generator_%06d.h5' % (epoch_no)
    filename3 = 'discriminator_%06d.h5' % (epoch_no)
    filename4 = 'gan_%06d.h5' % (epoch_no)

    g_model.save(filename2)
    d_model.save(filename3)
    gan_model.save(filename4)
    # shutil.copy(filename2, '/content/drive/MyDrive/navneet/'+filename2)
    # shutil.copy(filename3, '/content/drive/MyDrive/navneet/'+filename3)
    # shutil.copy(filename4, '/content/drive/MyDrive/navneet/'+filename4)
    print('>Saved: %s,%s,%s and %s' % (filename1, filename2, filename3, filename4))

def train(d_model, g_model, gan_model, dataset, dataset_val, n_epochs, n_batch=1):
  '''
  This function is for training the model based on the pixel-to-pixel architecture to achieve image to image tramslation using GAM
  args --> discriminator model, generator model, GAN model, train/test set, number of epochs, number of train batches/batch size
  '''
  # determine the output square shape of the discriminator
  n_patch = d_model.output_shape[1]
  # unpack dataset
  trainA, trainB = dataset
  # print(len(trainA))    2198 
  # calculate the number of batches per training epoch
  bat_per_epo = int(len(trainA) / n_batch)
  
  # calculate the number of training iterations
  n_steps = bat_per_epo * n_epochs
  # print(n_steps)   4396

  train_loss=[]
  val_loss=[]
  # manually enumerate epochs
  for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # [X_val_realA, X_val_realB], y_val_real = generate_real_samples(dataset_val, n_batch, n_patch)

        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # X_val_fakeB, y_val_fake = generate_fake_samples(g_model, X_val_realA, n_patch)

        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss1_log.append(d_loss1)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        d_loss2_log.append(d_loss2)
        
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        g_loss_log.append(g_loss)

        # summarize performance - plot loss per epoch
        psnr_before=psnr_metric(X_realA, X_realB) 
        psnr_after=psnr_metric(X_realB, X_fakeB)   # cloudy(X_realA)---> cloudless(X_realB)--- generated/predicted image (X_fakeB)
        ssim_before=ssim_metric(X_realA, X_realB)
        ssim_after=ssim_metric(X_realB, X_fakeB)
        mse_before=mean_squared_error(X_realA, X_realB)
        mse_after=mean_squared_error(X_realB, X_fakeB)
        
        psnr_before_log.append(psnr_before)
        psnr_after_log.append(psnr_after)
        ssim_before_log.append(ssim_before)
        ssim_after_log.append(ssim_after)
        mse_before_log.append(mse_before)
        mse_after_log.append(mse_after)
    
        # this will clear previous outputs
        # from IPython.display import clear_output
        # clear_output(wait=True)
        # display.display(plt.gcf())

        print('>%d, d1[%.3f] d2[%.3f] g[%.3f] psnr_before[%.3f] psnr_after[%.3f] ssim_before[%.3f] ssim_after[%.3f] mse_before[%.3f] mse_after[%.3f]' % (i+1, d_loss1, d_loss2, g_loss, psnr_before, psnr_after, ssim_before, ssim_after, mse_before, mse_after))
        
        if (i + 1) % 1000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        #for each step we will be checking train_loss and val_loss for early stopping (to avoid overfitting)
        train_loss.append((d_loss1+d_loss2+g_loss)/3) 
        val_loss.append((d_loss1 + d_loss2) / 2)
        if (i+1) % len(trainA) == 0:  
            epoch = (i+1)/len(trainA)
            averaged_psnr_before_values.append(np.mean(psnr_before_log))
            averaged_psnr_after_values.append(np.mean(psnr_after_log)) 
            averaged_ssim_before_values.append(np.mean(ssim_before_log))
            averaged_ssim_after_values.append(np.mean(ssim_after_log)) 
            averaged_mse_before_values.append(np.mean(mse_before_log))
            averaged_mse_after_values.append(np.mean(mse_after_log)) 
            averaged_dloss1_values.append(np.mean(d_loss1_log))
            averaged_dloss2_values.append(np.mean(d_loss2_log)) 
            averaged_gloss_values.append(np.mean(g_loss_log))

            workbook = xlsxwriter.Workbook("./results.xlsx")
            worksheet = workbook.add_worksheet()
            worksheet.write(0, 1, "PSNR_BEFORE")
            worksheet.write(0, 2, "PSNR_AFTER")
            worksheet.write(0, 3, "SSIM_BEFORE")
            worksheet.write(0, 4, "SSIM_AFTER")
            worksheet.write(0, 5, "MSE_BEFORE")
            worksheet.write(0, 6, "MSE_AFTER")
            worksheet.write(0, 7, "D_LOSS1")
            worksheet.write(0, 8, "D_LOSS2")
            worksheet.write(0, 9, "G_LOSS")
            worksheet.write(0, 10, "AVG_PSNR_BEFORE")
            worksheet.write(0, 11, "AVG_PSNR_AFTER")
            worksheet.write(0,12, "AVG_SSIM_BEFORE")
            worksheet.write(0,13, "AVG_SSIM_AFTER")
            worksheet.write(0, 14, "AVG_MSE_BEFORE")
            worksheet.write(0, 15, "AVG_MSE_AFTER")
            worksheet.write(0, 16, "AVG_DLOSS1")
            worksheet.write(0, 17, "AVG_DLOSS2")
            worksheet.write(0, 18, "AVG_GLOSS")

            worksheet.write_column(1,1, averaged_psnr_before_values)
            worksheet.write_column(1,2, averaged_psnr_after_values) 
            worksheet.write_column(1,3, averaged_ssim_before_values)
            worksheet.write_column(1,4, averaged_ssim_after_values)
            worksheet.write_column(1,5, averaged_mse_before_values)
            worksheet.write_column(1,6, averaged_mse_after_values)
            worksheet.write_column(1,7, averaged_dloss1_values)
            worksheet.write_column(1,8, averaged_dloss2_values)
            worksheet.write_column(1,9, averaged_gloss_values)

            psnr_before_log.clear()
            psnr_after_log.clear()
            ssim_before_log.clear()
            ssim_after_log.clear()
            mse_before_log.clear()
            mse_after_log.clear()
            d_loss1_log.clear()
            d_loss2_log.clear()
            g_loss_log.clear()
            
            plt.plot(averaged_psnr_before_values)
            plt.plot(averaged_psnr_after_values)
            plt.title('model psnr - metric')
            plt.ylabel('psnr')
            plt.xlabel('epoch')
            plt.legend(['before psnr', 'after psnr'], loc='lower right')
            plt.savefig('./psnr_ae_echo')
            # plt.show()

            plt.plot(averaged_ssim_before_values)
            plt.plot(averaged_ssim_after_values)
            plt.title('model ssim - metric')
            plt.ylabel('ssim')
            plt.xlabel('epoch')
            plt.legend(['before ssim', 'after ssim'], loc='lower right')
            plt.savefig('./ssim_ae_echo')

            plt.plot(averaged_mse_before_values)
            plt.plot(averaged_mse_after_values)
            plt.title('model mse - metric')
            plt.ylabel('mse')
            plt.xlabel('epoch')
            plt.legend(['before mse', 'after mse'], loc='lower right')
            plt.savefig('./mse_ae_echo')

            psnr_before_avg=np.mean(averaged_psnr_before_values)
            psnr_after_avg=np.mean(averaged_psnr_after_values)
            print('average psnr before',psnr_before_avg)
            print('average psnr after',psnr_after_avg)

            ssim_before_avg=np.mean(averaged_ssim_before_values)
            ssim_after_avg=np.mean(averaged_ssim_after_values)
            print('average ssim before',ssim_before_avg)
            print('average ssim after',ssim_after_avg)

            mse_before_avg=np.mean(averaged_mse_before_values)
            mse_after_avg=np.mean(averaged_mse_after_values)
            print('average mse before',mse_before_avg)
            print('average mse after',mse_after_avg)

            d_loss1_avg=np.mean(averaged_dloss1_values)
            d_loss2_avg=np.mean(averaged_dloss2_values)
            g_loss_avg=np.mean(averaged_gloss_values)

            worksheet.write(1, 10, psnr_before_avg)
            worksheet.write(1, 11, psnr_after_avg)
            worksheet.write(1, 12, ssim_before_avg)
            worksheet.write(1, 13, ssim_after_avg)
            worksheet.write(1, 14, mse_before_avg)
            worksheet.write(1, 15, mse_after_avg)
            worksheet.write(1, 16, d_loss1_avg)
            worksheet.write(1, 17, d_loss2_avg)
            worksheet.write(1, 18, g_loss_avg)
            workbook.close()

            # # Update the EarlyStopping callback with the current validation loss
            # early_stopping.on_epoch_end(epoch, {'val_loss': val_loss})

            # #Check if training should stop
            # if early_stopping.stopped_epoch is not None:
            #     print(f"Training stopped at epoch {early_stopping.stopped_epoch} because validation loss did not improve.")
            #     break

            # train_losses.append(np.mean(train_loss))
            # val_losses.append(np.mean(val_loss))
            # print(f"Epoch {epoch}: train loss = {np.mean(train_loss):.4f}, val loss = {np.mean(val_loss):.4f}")
            # train_loss=[]
            # val_loss=[]

        #it will execute after 10 epochs so that generator and discriminator are saved and loaded
        if (i+1) % (len(trainA) * 10) == 0:
            plt.clf()
            plt.figure(figsize=(20,12))
            plt.title('Epoch:%d, d1[%.3f] d2[%.3f] g[%.3f]' % (n_epochs, d_loss1, d_loss2, g_loss))
            plt.xlabel('No of iterations', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.plot(averaged_dloss1_values, 'r-', lw=2, label='d_loss1')
            plt.plot(averaged_dloss2_values, 'b-', lw=1, label='d_loss2')
            plt.plot(averaged_gloss_values, 'g-', lw=1, label='g_loss')
            plt.legend(prop={'size':16}, loc="center")
            plt.show()
            plt.savefig('./loss_graph_%06d.jpg' % (i+1), bbox_inches='tight')
            summarize_performance(i, g_model, d_model, gan_model, dataset, len(trainA))

def load_images(path, size=(256,256)):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path+'cloudless'):
        pixels_src = load_img(path + 'cloudy/' + filename, target_size=size)
        pixels_tar = load_img(path + 'cloudless/' + filename, target_size=size)
        # convert to numpy array
        pixels_src = img_to_array(pixels_src)
        pixels_tar = img_to_array(pixels_tar)

        cloudless_img=pixels_tar[:, :256]
        cloudy_img=pixels_src[:, :256]
        src_list.append(cloudy_img)
        tar_list.append(cloudless_img)
    return [asarray(src_list), asarray(tar_list)]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()

# def cal_avg(log, count):
#     avg_log=[]
#     for i in range(0, len(log), count):
#         start = i
#         end = i + count
#         window = log[start:end]
#         average = sum(window) / len(window)
#         avg_log.append(average)
#     return avg_log

if __name__ == '__main__':
    desired_size = (256, 256)
    for filename in os.listdir("/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/cloudless"):
        img = Image.open(os.path.join("/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/cloudless", filename))
        old_size = img.size
        new_size = desired_size
        new_img = img.resize(new_size)
        new_img.save(os.path.join("/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/cloudless", filename))
    for filename in os.listdir("/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/cloudy"):
        img = Image.open(os.path.join("/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/cloudy", filename))
        old_size = img.size
        new_size = desired_size
        new_img = img.resize(new_size)
        new_img.save(os.path.join("/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/cloudy", filename))
    test_discr = define_discriminator(in_shape=(256,256,3))
    print(test_discr.summary())
    tf.keras.utils.plot_model(test_discr, show_shapes=True, dpi=64)
    test_gen = define_generator(image_shape=(256,256,3))
    print(test_gen.summary())
    tf.keras.utils.plot_model(test_gen, show_shapes=True, dpi=64)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(test_gen=test_gen,
                                 test_discr=test_discr)
    input_folder = '/workplace/OpticalRemoteSensingClassification/paired_dataset/ShipRSImageNet_V1/VOC_Format/'

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    #Train, val, test
    splitfolders.ratio(input_folder, output="dataset", 
                    seed=42, ratio=(.8, .2), 
                    group_prefix=None) # default values
    src='dataset/val'
    dest='dataset/test'
    os.rename(src,dest)

    # dataset path
    path = '/workplace/OpticalRemoteSensingClassification/paired_dataset/dataset/train/'
    # path_val= 'dataset/images/val/'
    path_test= '/workplace/OpticalRemoteSensingClassification/paired_dataset/dataset/test/'
    # load dataset
    [src_images, tar_images] = load_images(path)
    # [src_val_images, tar_val_images] = load_images(path_val)
    [src_test_images, tar_test_images] = load_images(path_test)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # print('Loaded: ', src_val_images.shape, tar_val_images.shape)
    print('Loaded: ', src_test_images.shape, tar_test_images.shape)
    # define input shape based on the loaded dataset
    image_shape = src_images.shape[1:]
    # define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    # define the composite model
    gan_model = define_gan(g_model, d_model, image_shape)

    # code to load models so that we can train our model in batches and save
    # last_epoch = 10
    # g_model = keras.models.load_model('/content/drive/MyDrive/navneet/generator_000010.h5')
    # d_model = keras.models.load_model('/content/drive/MyDrive/navneet/discriminator_000010.h5')
    # gan_model = keras.models.load_model('/content/drive/MyDrive/navneet/gan_000010.h5')


    data = [src_images, tar_images]
    # load and prepare test images
    # data_val = [src_val_images, tar_val_images]
    data_test = [src_test_images, tar_test_images]
    dataset = preprocess_data(data)
    print(dataset[0].shape, dataset[1].shape)
    # dataset_val = preprocess_data(data_val)
    dataset_test = preprocess_data(data_test)
    start1 = datetime.now() 
    n_epochs=100
    train(d_model, g_model, gan_model, dataset, dataset_test, n_epochs, n_batch=1) 
    countTrain = 2198
    stop1 = datetime.now()
    #Execution time of the model 
    execution_time = stop1-start1
    print("Execution time is: ", execution_time)
    model = load_model('generator_100.h5')
    [X1, X2] = dataset
    # select random example
    ix = randint(0, len(X1), 1)
    src_image, tar_image = X1[ix], X2[ix]
    # generate image from source
    gen_image = model.predict(src_image)
    # plot all three images
    plot_images(src_image, gen_image, tar_image)


    #plot images of test dataset
    [Y1, Y2] = dataset_test
    # select random example
    iy = randint(0, len(Y1), 1)
    src_test_image, tar_test_image = Y1[iy], Y2[iy]
    # generate image from source
    gen_test_image = model.predict(src_test_image)
    # plot all three images
    plot_images(src_test_image, gen_test_image, tar_test_image)
    # averaged_psnr_before_values = []
    # averaged_psnr_after_values = []
    # countTrain = len(os.listdir('/workplace/OpticalRemoteSensingClassification/paired_dataset/dataset/train/cloudy'))
    # averaged_psnr_before_values = cal_avg(psnr_before_log, countTrain)
    # averaged_psnr_after_values = cal_avg(psnr_after_log, countTrain)

    # plt.plot(averaged_psnr_before_values)
    # plt.plot(averaged_psnr_after_values)
    # plt.title('model psnr - metric')
    # plt.ylabel('psnr')
    # plt.xlabel('epoch')
    # plt.legend(['Before psnr', 'After psnr'], loc='lower right')
    # plt.savefig('./psnr.jpg', bbox_inches='tight')
    # # plt.savefig('/workplace/OpticalRemoteSensingClassification/paired_dataset/psnr')
    # plt.show()
    # averaged_ssim_before_values = []
    # averaged_ssim_after_values = []
    # averaged_ssim_before_values = cal_avg(ssim_before_log, countTrain)
    # averaged_ssim_after_values = cal_avg(ssim_after_log, countTrain)

    # plt.plot(averaged_ssim_before_values)
    # plt.plot(averaged_ssim_after_values)
    # plt.title('model ssim - metric')
    # plt.ylabel('ssim')
    # plt.xlabel('epoch')
    # plt.legend(['Before ssim', 'After ssim'], loc='lower right')
    # plt.savefig('./ssim.jpg', bbox_inches='tight')
    # # plt.savefig('/workplace/OpticalRemoteSensingClassification/paired_dataset/ssim')
    # plt.show()

    # averaged_mse_before_values = []
    # averaged_mse_after_values = []
    # averaged_mse_before_values = cal_avg(mse_before_log, countTrain)
    # averaged_mse_after_values = cal_avg(mse_after_log, countTrain)  

    # plt.plot(averaged_mse_before_values)
    # plt.plot(averaged_mse_after_values)
    # plt.title('model mse - metric')
    # plt.ylabel('mse')
    # plt.xlabel('epoch')
    # plt.legend(['Before mse','After mse'], loc='lower right')
    # plt.savefig('./mse.jpg', bbox_inches='tight')
    # # plt.savefig('/workplace/OpticalRemoteSensingClassification/paired_dataset/mse')
    # plt.show()

    # #calculating average psnr
    # psnr_before_avg=sum(psnr_before_log)/len(psnr_before_log)
    # print('average psnr',psnr_before_avg)
    # psnr_after_avg=sum(psnr_after_log)/len(psnr_after_log)
    # print('average psnr after',psnr_after_avg)

    # ssim_before_avg=sum(ssim_before_log)/len(ssim_before_log)
    # print('average ssim',ssim_before_avg.numpy())
    # ssim_after_avg=sum(ssim_after_log)/len(ssim_after_log)
    # print('average ssim after',ssim_after_avg.numpy())

    # mse_before_avg=sum(mse_before_log)/len(mse_before_log)
    # print('average mse',mse_before_avg.numpy())
    # mse_after_avg=sum(mse_after_log)/len(mse_after_log)
    # print('average mse after',mse_after_avg.numpy())

    # d_loss1_avg=sum(d_loss1_log)/len(d_loss1_log)
    # print('average dloss1', d_loss1_avg)

    # d_loss2_avg=sum(d_loss2_log)/len(d_loss2_log)
    # print('average dloss2',d_loss2_avg)

    # g_loss_avg=sum(g_loss_log)/len(g_loss_log)
    # print('average gloss',g_loss_avg)
    # workbook = xlsxwriter.Workbook("/workplace/OpticalRemoteSensingClassification/paired_dataset/results.xlsx")
    # worksheet = workbook.add_worksheet()

    # worksheet.write(0, 1, "D_LOSS1")

    # worksheet.write_column(1,1, d_loss1_log)

    # worksheet.write(0, 2, "D_LOSS2")

    # worksheet.write_column(1,2, d_loss2_log)

    # worksheet.write(0, 3, "G_LOSS")

    # worksheet.write_column(1,3, g_loss_log)

    # worksheet.write(0, 4, "PSNR_BEFORE")

    # worksheet.write_column(1,4, psnr_before_log)

    # worksheet.write(0, 5, "AVG_PSNR_BEFORE")

    # worksheet.write(1, 5, psnr_before_avg)

    # worksheet.write(0, 6, "PSNR_AFTER")

    # worksheet.write_column(1,6, psnr_after_log)

    # worksheet.write(0, 7, "AVG_PSNR_AFTER")

    # worksheet.write(1, 7, psnr_after_avg)

    # worksheet.write(0, 8, "SSIM_BEFORE")

    # worksheet.write_column(1,8, ssim_before_log)

    # worksheet.write(0,9, "AVG_SSIM_BEFORE")

    # worksheet.write(1, 9, ssim_before_avg)

    # worksheet.write(0, 10, "SSIM_AFTER")

    # worksheet.write_column(1,10, ssim_after_log)

    # worksheet.write(0,11, "AVG_SSIM_AFTER")

    # worksheet.write(1, 11, ssim_after_avg)

    # worksheet.write(0, 12, "MSE_BEFORE")

    # worksheet.write_column(1,12, mse_before_log)

    # worksheet.write(0, 13, "AVG_MSE_BEFORE")

    # worksheet.write(1, 13, mse_before_avg)

    # worksheet.write(0, 14, "MSE_AFTER")

    # worksheet.write_column(1,14, mse_after_log)

    # worksheet.write(0, 15, "AVG_MSE_AFTER")

    # worksheet.write(1, 15, mse_after_avg)

    # worksheet.write(0, 16, "AVG_DLOSS1")

    # worksheet.write(1, 16, d_loss1_avg)

    # worksheet.write(0, 17, "AVG_DLOSS2")

    # worksheet.write(1, 17, d_loss2_avg)

    # worksheet.write(0, 18, "AVG_GLOSS")

    # worksheet.write(1, 18, g_loss_avg)

    # workbook.close()

