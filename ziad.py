from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,MaxPooling1D,Convolution1D,ZeroPadding1D
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
import numpy as np
from numpy import zeros, newaxis

import os


from sklearn.cross_validation import train_test_split

def load_data_from_csv_wordCount(sizeOfDataSet=0, file=None):
    # pass 0 in sizeOfDataSet to get full dataset
    file = open(file, 'r')
    data = []
    dict={}
    cls = []
    index = -1
    for video in file:

        if (sizeOfDataSet != 0):
            if (len(dict) == sizeOfDataSet+1):
                break;

        if index == -1:
            index += 1
            continue;

        features = video.split(',')
        # if len(features)<580 or len(features) >590:
        #     continue
        if features[1101] not in dict:
            dict.update({features[1101]:1})
        # elif dict[features[581] ]==100:
        #     continue;
        # else :
        #     dict[features[581]]+=1
        # # print(features[581] != 'BANKS')
        data.append([])

        # print(1)

        for i in range(1, 1101):
            data[index].append(features[i])
            if (i == 1100):
                cls.append(features[1101][:-1])
                # cls[len(cls)-1]=cls[len(cls)-1][:-1]
        index += 1
    # print('why')
    if sizeOfDataSet !=0:
        dict.popitem()
        data=data[:len(data)-1]
        cls = cls[:len(cls) - 1]
    return np.array(data), np.array(cls),dict

path1 = "ourInterpolate.csv"

dataSetSize = 4
x=[]
y=[]

#x = features
#y = labels
#z = dictionary
x,y,z = load_data_from_csv_wordCount(dataSetSize,path1)

vectorOfPoints =[]

vectorOfPoints = x

# x= x[...,newaxis]
classesNumber = dataSetSize
batchSize = 20
epochNumber = 40
convNumber = 1

vectorOfPoints=np.reshape(vectorOfPoints,(len(vectorOfPoints),110,10))

x_train, x_test, y_train, y_test = train_test_split(vectorOfPoints,y,test_size = 0.06,random_state = 613)

uniques, trainID= np.unique(y_train, False,True)
y_train = np_utils.to_categorical(trainID,classesNumber)
print(y_train)

uniques, trainID= np.unique(y_test, False,True)
y_test= np_utils.to_categorical(trainID,classesNumber)


print("1")


model = Sequential()
model.add(Convolution1D(8, convNumber, activation="relu", name="conv1_1",input_shape=(110,10)))
model.add(Convolution1D(8, convNumber, activation='relu', name='conv1_2'))
#
model.add(Convolution1D(16, convNumber, activation='relu', name='conv2_1'))
model.add(Convolution1D(16, convNumber, activation='relu', name='conv2_2'))

model.add(Convolution1D(32, convNumber, activation='relu', name='conv3_1'))
model.add(Convolution1D(32, convNumber, activation='relu', name='conv3_2'))
model.add(Convolution1D(32, convNumber, activation='relu', name='conv3_3'))

model.add(Convolution1D(64, convNumber, activation='relu', name='conv4_1'))
model.add(Convolution1D(64, convNumber, activation='relu', name='conv4_2'))
model.add(Convolution1D(64, convNumber, activation='relu', name='conv4_3'))

model.add(Convolution1D(64, convNumber, activation='relu', name='conv5_1'))
model.add(Convolution1D(64, convNumber, activation='relu', name='conv5_2'))
model.add(Convolution1D(64, convNumber, activation='relu', name='conv5_3'))

top_model = Sequential()

#Fully connected layers
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(1024, activation='sigmoid'))
top_model.add(Dense(1024, activation='sigmoid'))
top_model.add(Dense(classesNumber, activation='softmax'))

# add the model on top of the convolutional base
model.add(top_model)
print("2")

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

# for layers in model.layers[:-1]:
#     layers.trainable = False

#model = load_model('saved models/weights.10.h5')

filepath="saved modelss/weights.{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]


# model.fit(x_train,y_train,batch_size = batchSize, validation_data=(x_test,y_test), nb_epoch = epochNumber,verbose =1,callbacks = callbacks_list)

import conf
from keras.models import load_model
Model=load_model("saved modelss/weights.28.h5")
conf.cnfu_matrix(uniques,Model,x_test,y_test,"CNN")
# print(counter)
#joblib.dump(model, 'clf.pkl')