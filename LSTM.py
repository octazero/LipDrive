from keras.models import Sequential,load_model
from keras.layers import LSTM ,Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
import operator
from keras.callbacks import EarlyStopping,ModelCheckpoint
from JSON import load_data_from_csv_word_inDiction,load_data_from_csv_wordCount
import matplotlib.pyplot as plt
import numpy as np

x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=5,file="Geometric.csv")
xtest,ytest=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="test_dataset.csv",dict=d)
xv,yv=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="validate_dataset.csv",dict=d)
x=x.reshape(len(x),20,29)
xtest=xtest.reshape(len(xtest),20,29)
xv=xv.reshape(len(xv),20,29)
y.sort()
ytest.sort()
yv.sort()
uniques, id_test1=np.unique(y,return_inverse=True)
y=to_categorical(id_test1,5)

uniques, id_test=np.unique(ytest,return_inverse=True)
ytest=to_categorical(id_test,5)


uniques, id_test=np.unique(yv,return_inverse=True)
yv=to_categorical(id_test,5)
sd=SGD(lr=0.00001)
ES=EarlyStopping(patience=2)
check=ModelCheckpoint(filepath="model5.h5",save_best_only=True,period=1,mode="min")
Model=Sequential()
Model.add(LSTM(47,input_shape=(20,29),return_sequences=True))
Model.add(LSTM(32,activation='relu',return_sequences=True))
Model.add(LSTM(256,activation='relu',return_sequences=True))
Model.add(LSTM(512,activation='relu',return_sequences=True))
Model.add(LSTM(256,activation='relu',return_sequences=True))
Model.add(LSTM(128,activation='relu',return_sequences=True))
Model.add(LSTM(64,activation='relu',return_sequences=True))
Model.add(LSTM(32,activation='relu'))
Model.add(Dense(5,activation='sigmoid'))
Model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
train_result=Model.fit(x,y,validation_data=(xv,yv),epochs=40,callbacks=[check])
Model.save("hello1.h5")
# plt.plot(train_result.history['val_loss'], 'r')
# plt.xlabel('Epochs')
# plt.ylabel('Validation score')
# plt.show()
# Model=load_model("model1.h5")
# print(Model.summary())

print(Model.predict(np.array([xtest[120]])))
print("-----------------------------------------------------------------------------------")
print(ytest[120])
#
print("-----------------------------------------------------------------------------------")
# print(Model.evaluate(xtest,ytest))
counter=[0,0,0,0,0]
for i in range (len(xtest)):
    result=Model.predict(np.array([xtest[i]]))
    index, value = max(enumerate(result[0]), key=operator.itemgetter(1))
    index1, value1 = max(enumerate(ytest[i]), key=operator.itemgetter(1))
    if index==index1:
        counter[index] +=1

print(counter)
