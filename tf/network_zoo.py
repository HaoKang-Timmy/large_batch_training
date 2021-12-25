from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
 
img_size = (32, 32, 3)


def kerasnet(nb_classes):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, padding='valid',
                                input_shape=(3,32,32)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3, padding='valid'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(mode=2))
        model.add(Activation('relu'))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        return model

def shallownet(nb_classes):
	global img_size
	model = Sequential()
	model.add(Convolution2D(64, 5, 5, padding='same', input_shape=img_size))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
	
	model.add(Convolution2D(64, 5, 5, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

	model.add(Flatten())
	model.add(Dense(384))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5)) 
	model.add(Dense(192))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5)) 
	model.add(Dense(nb_classes, activation='softmax'))
	return model


def deepnet(nb_classes):
	global img_size
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, padding='same', input_shape=img_size))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.3))
	model.add(Convolution2D(64, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
	model.add(Convolution2D(128, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(128, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
	model.add(Convolution2D(256, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(256, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(256, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
	model.add(Convolution2D(512, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(512, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(512, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
	model.add(Convolution2D(512, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(512, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu')); model.add(Dropout(0.4))
	model.add(Convolution2D(512, 3, 3, padding='same'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
	model.add(Flatten()); model.add(Dropout(0.5))
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu')); model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))
	return model

