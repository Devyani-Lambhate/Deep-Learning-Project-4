import tensorflow as tf
# example of a dcgan on cifar10
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.convolutional import _Conv
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Concatenate
#from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D

from keras.legacy import interfaces
from keras.engine import InputSpec
import keras.backend as K


class SNConv2D(_Conv):
    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.input_spec = InputSpec(ndim=4)
        self.Ip = 1
        self.u = self.add_weight(
            name='W_u',
            shape=(1,filters),
            initializer='random_uniform',
            trainable=False
        )

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.W_bar(),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(SNConv2D, self).get_config()
        config.pop('rank')
        return config

    def W_bar(self):
        # Spectrally Normalized Weight
        W_mat = K.permute_dimensions(self.kernel, (3, 2, 0, 1)) # (h, w, i, o) => (o, i, h, w)
        W_mat = K.reshape(W_mat,[K.shape(W_mat)[0], -1]) # (o, i * h * w)

        if not self.Ip >= 1:
            raise ValueError("The number of power iterations should be positive integer")

        _u = self.u
        _v = None

        for _ in range(self.Ip):
            _v = _l2normalize(K.dot(_u, W_mat))
            _u = _l2normalize(K.dot(_v, K.transpose(W_mat)))

        sigma = K.sum(K.dot(_u,W_mat)*_v)

        K.update(self.u,K.in_train_phase(_u, self.u))
        return self.kernel / sigma

def _l2normalize(x):
    return x / K.sqrt(K.sum(K.square(x)) + K.epsilon())

# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3),n_classes=10):
	in_label=Input(shape=(1,))
	li=Embedding(n_classes,50)(in_label)
	n_nodes=in_shape[0]*in_shape[1]*3
	li=Dense(n_nodes)(li)
	li=Reshape((in_shape[0],in_shape[1],3))(li)
	in_image=Input(shape=in_shape)

	merge=Concatenate()([in_image,li])
	
	#model = Sequential()
	# normal
	fe=SNConv2D(64, (3,3), padding='same', input_shape=in_shape)(merge)
	fe=LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe=SNConv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe=LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe=SNConv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe=LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe=SNConv2D(256, (3,3), strides=(2,2), padding='same')(fe)
	fe=LeakyReLU(alpha=0.2)(fe)
	# classifier
	fe=Flatten()(fe)
	fe=Dropout(0.4)(fe)
	out_layer=Dense(1, activation='sigmoid')(fe)
	model=Model([in_image,in_label],out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim,n_classes=10):
	in_label=Input(shape=(1,))
	li=Embedding(n_classes,50)(in_label)
	n_nodes=4*4*3
	li=Dense(n_nodes)(li)
	li=Reshape((4,4,3))(li)
	in_lat=Input(shape=(latent_dim,))
	
	#model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	gen=Dense(n_nodes)(in_lat)
	gen=LeakyReLU(alpha=0.2)(gen)
	gen=Reshape((4, 4, 256))(gen)

	merge=Concatenate()([gen,li])
	# upsample to 8x8
	gen=Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen=LeakyReLU(alpha=0.2)(gen)
	# upsample to 16x16
	gen=Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen=LeakyReLU(alpha=0.2)(gen)
	# upsample to 32x32
	gen=Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen=LeakyReLU(alpha=0.2)(gen)
	# output layer
	out_layer= Conv2D(3, (3,3), activation='tanh', padding='same')(gen)
	model=Model([in_lat ,in_label], out_layer)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	#model = Sequential()
	gen_noise,gen_label=g_model.input
	gen_output=g_model.output
	gan_output= d_model([gen_output,gen_label])
	model=Model([gen_noise,gen_label], gan_output)	
	# add generator
	#model.add(g_model)
	# add the discriminator
	#model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# load and prepare cifar10 training images
def load_real_samples():
	# load cifar10 dataset
	(trainX, trainY), (_, _) = load_data()
	# convert from unsigned ints to floats
	X = trainX.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return [X,trainY]
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	images, labels=dataset
	
	ix = randint(0, images.shape[0], n_samples)
	# retrieve selected images
	X,labels = images[ix], labels[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return [X,labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	z_input=x_input.reshape(n_samples,latent_dim)	
	labels=randint(0,n_classes,n_samples)
	# reshape into a batch of inputs for the network
	#x_input = x_input.reshape(n_samples, latent_dim)
	return [z_input,labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	z_input,labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict([z_input,labels_input])
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return [X,labels_input], y
 
# create and save a plot of generated images
def save_plot(examples, epoch, n=8):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	[X_real,real_label], y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate([X_real,real_label], y_real, verbose=0)
	# prepare fake examples
	[x_fake,fake_label], y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate([x_fake,fake_label], y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)
 


import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
#from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model_in, images1, images2):
	# calculate activations
	act1 = model_in.predict(images1)
	act2 = model_in.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 



model_in = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))



# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	itr=0
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			itr+=1
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake ,labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake,labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input,labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		
		#labels1=numpy.where(labels_real==2)
		#print(labels1)
		#labels2=numpy.where(labels==2)
		#print(labels2)
		#images1 = X_real[numpy.array(labels1[0])]
		#images2 = X_fake[numpy.array(labels2[0])]
			if(itr%1000==0):
				#images1=X_real
				# prepare the inception v3 model
				# load cifar10 images
				(_, _), (images1 , _) = cifar10.load_data()
				shuffle(images1)
				images1=images1[:1000]
				[x_fake,fake_label], y_fake = generate_fake_samples(g_model, latent_dim, n_samples=1000)
				images2=x_fake
				images2=(images2+1)/2
				images2=255*images2			
				
				print('Loaded', images1.shape, images2.shape)
				# convert integer to floating point values
				images1 = images1.astype('float32')
				images2 = images2.astype('float32')
				# resize images
				images1 = scale_images(images1, (299,299,3))
				images2 = scale_images(images2, (299,299,3))
				print('Scaled', images1.shape, images2.shape)
				# pre-process images
				images1 = preprocess_input(images1)
				images2 = preprocess_input(images2)
				# calculate fid
				fid = calculate_fid(model_in, images1, images2)
				print('FID: %.3f' % fid)
				fid_score.append(fid)
				epoch_arr.append(i)
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

fid_score=[]
epoch_arr=[]
n_classes=10
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()




# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

pyplot.plot(epoch_arr,fid_score)
pyplot.savefig('fid_score')
pyplot.close()

from keras.utils import plot_model
plot_model(gan_model, to_file='dc_model_plot.png', show_shapes=True, show_layer_names=True)

