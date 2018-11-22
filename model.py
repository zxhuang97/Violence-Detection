from keras.models import Sequential,Model
from keras.layers import Input,Maximum,Dense, Dropout, Flatten, TimeDistributed,GlobalMaxPool1D,BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
import glob
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
def get_c3d():
    """ Return the Keras model of the network
    """
    model = Sequential()

    input_shape=(16, 112, 112, 3) # l, h, w, c

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))
    model.load_weights("../c3d-keras/models/c3d.h5")

    input = model.layers[0].input
    output = model.layers[-6].output
    output = Dense(1024, activation='relu', name='fc6')(output)
    output = Dropout(.5)(output)
    output = Dense(1024, activation='relu', name='fc7')(output)
    output = Dropout(.5)(output)
    output = Dense(1, activation='sigmoid', name='fc8')(output)

    return Model(input,output)

def build_model():
    input= Input((32,16,112,112,3))
    x = BatchNormalization()(input)
    c3d = get_c3d()
    x = TimeDistributed(c3d)(x)
    x = GlobalMaxPool1D()(x)
    model = Model(input,x)
    model.summary()
    return model



build_model()