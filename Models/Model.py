
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential, Model
from keras.layers import LSTM, Dropout, Concatenate, Input, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Conv2D
from keras.layers import concatenate
import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras import backend as K
from keras.utils import plot_model

class Model(Model):

    def __init__(self):
        super(Model, self).__init__()

        input_load = Input(shape=(time_steps,1), name="input_load")

        lstm1 = LSTM(units=time_steps, return_sequences=True)(input_load)
        lstm2 = LSTM(units=time_steps)(lstm1)
        dropout = Dropout(rate=0.2)(lstm2)

        dense = Dense(units=1, activation="linear")(dropout)

        model = Model(inputs=input_load, outputs=dense)
        
        # Layer of Block 1
        self.conv1 = Conv2D(32, 3, strides=2, activation="relu")
        self.max1  = MaxPooling2D(3)
        self.bn1   = BatchNormalization()

        # Layer of Block 2
        self.conv2 = Conv2D(64, 3, activation="relu")
        self.bn2   = BatchNormalization()
        self.drop  = Dropout(0.3)

        # GAP, followed by Classifier
        self.gap   = GlobalAveragePooling2D()
        self.dense = Dense(num_classes)


    def call(self, input_tensor, training=False):
        # forward pass: block 1 
        x = self.conv1(input_tensor)
        x = self.max1(x)
        x = self.bn1(x)

        # forward pass: block 2 
        x = self.conv2(x)
        x = self.bn2(x)

        # droput followed by gap and classifier
        x = self.drop(x)
        x = self.gap(x)
        return self.dense(x)