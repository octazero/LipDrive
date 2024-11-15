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


from sklearn.model_selection import train_test_split

def unpart1by1(n):
    n &= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n

def deinterleave2(n):
    return unpart1by1(n), unpart1by1(n >> 1)

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
        if features[581] not in dict:
            dict.update({features[581]:1})
        # elif dict[features[581] ]==100:
        #     continue;
        # else :
        #     dict[features[581]]+=1
        # # print(features[581] != 'BANKS')
        data.append([])

        # print(1)

        for i in range(1, 581):
            data[index].append(int(features[i]))
            if (i == 580):
                cls.append(features[581][:-1])
                # cls[len(cls)-1]=cls[len(cls)-1][:-1]
        index += 1
    # print('why')
    if sizeOfDataSet !=0:
        dict.popitem()
        data=data[:len(data)-1]
        cls = cls[:len(cls) - 1]
    return np.array(data), np.array(cls),dict

path1 = "data.csv"

dataSetSize = 5
x=[]
y=[]

#x = features
#y = labels
#z = dictiona
x,y,z = load_data_from_csv_wordCount(dataSetSize,path1)

vectorOfPoints =[]
x1,y1 = deinterleave2(x[0][0])


for i in x:
    Points = []
    for j in i:
        newx, newy = deinterleave2(j)
        Points.append([newx,newy,0])
    vectorOfPoints.append(Points)


# x= x[...,newaxis]
print(len(vectorOfPoints))

classesNumber = dataSetSize
batchSize = 20
epochNumber = 40
poolNumber = 2
convNumber = 1
vectorOfPoints=np.reshape(vectorOfPoints,(len(vectorOfPoints),29,20,3))
print(vectorOfPoints.shape)

x_train, x_test, y_train, y_test = train_test_split(vectorOfPoints,y,test_size = 0.2,random_state = 4)
uniques, trainID= np.unique(y_train, False,True)
y_train = np_utils.to_categorical(trainID,classesNumber)


uniques, trainID= np.unique(y_test, False,True)
y_test = np_utils.to_categorical(trainID,classesNumber)

x_train=np.reshape(x_train,(len(x_train),29,20,3))
x_test = np.reshape(x_test,(len(x_test),29,20,3))

model = Sequential()
model.add(Convolution2D(64, (convNumber,convNumber), activation="relu", name="conv1_1",input_shape=(29,20,3)))
model.add(Convolution2D(64, (convNumber,convNumber), activation="relu", name="conv1_2"))
# model.add(MaxPooling2D((poolNumber,poolNumber), strides=(2,2)))

model.add(Convolution2D(128, (convNumber,convNumber), activation='relu', name='conv2_1'))
model.add(Convolution2D(128, (convNumber,convNumber), activation='relu', name='conv2_2'))
# model.add(MaxPooling2D((poolNumber,poolNumber), strides=(2,2)))

model.add(Convolution2D(256, (convNumber,convNumber), activation='relu', name='conv3_1'))
model.add(Convolution2D(256, (convNumber,convNumber), activation='relu', name='conv3_2'))
model.add(Convolution2D(256, (convNumber,convNumber), activation='relu', name='conv3_3'))
# model.add(MaxPooling2D((poolNumber,poolNumber), strides=(2,2)))

model.add(Convolution2D(512, (convNumber,convNumber), activation='relu', name='conv4_1'))
model.add(Convolution2D(512, (convNumber,convNumber), activation='relu', name='conv4_2'))
model.add(Convolution2D(512, (convNumber,convNumber), activation='relu', name='conv4_3'))
# model.add(MaxPooling2D((poolNumber,poolNumber), strides=(2,2)))

model.add(Convolution2D(512, (convNumber,convNumber), activation='relu', name='conv5_1'))
model.add(Convolution2D(512, (convNumber,convNumber), activation='relu', name='conv5_2'))
model.add(Convolution2D(512, (convNumber,convNumber), activation='relu', name='conv5_3'))
# model.add(MaxPooling2D((poolNumber,poolNumber), strides=(2,2)))



top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
# with l2 regularizer
#, W_regularizer=l2(0.1))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dense(classesNumber, activation='softmax'))

# add the model on top of the convolutional base
model.add(top_model)


model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

for layers in model.layers[:-1]:
    layers.trainable = False

#model = load_model('saved models/weights.40.h5')

filepath="model/weights.{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]


model.fit(x_train,y_train,batch_size = batchSize, nb_epoch = epochNumber,verbose =1,callbacks = callbacks_list, validation_data=(x_test,y_test) )

#joblib.dump(model, 'clf.pkl')

