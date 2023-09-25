from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
# one hot encode outputs
from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
from keras.constraints import MaxNorm
import warnings
import logging, os
from Main.metrics import metric

#from Main import Visualize
#from keras.utils import plot_model

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
import numpy as np


def cnn(X,Y, tr,MSE, RMSE, R_sq, MAE):
    X_train, X_test, y_train, y_test = train_test_split(X+Y, Y, train_size=tr, random_state=42)
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train)
    X_test = (X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_classes = 1

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(1, X_train.shape[1],1), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=MaxNorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, kernel_constraint=MaxNorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    epochs = 1
    optimizer = 'adam'
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()
    X_train = np.resize(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
    X_test = np.resize(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, verbose=0)
    y_pred = model.predict(X_test)
    y_test = np.resize(y_test, (len(y_test),))
    mse, rmse, r, mae = metric(y_test,y_pred)

    MSE.append(mse)
    RMSE.append(rmse)
    R_sq.append(r)
    MAE.append(mae)

