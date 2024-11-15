import json
# from keras.models import Sequential,load_model
# from keras.layers import Input, LSTM, Dense,Embedding
# import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from os import walk
from sklearn import svm
from sklearn.linear_model import  LogisticRegression
import numpy as np
import collections
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import itertools
# Model=Sequential()
# grid=joblib.load('gridsearch.pkl')
#
# print(grid.scorer_)

Modelss={}

def exportConfusion(cls,matrix):
    arr=[]
    file=open('confusionMat','w')
    for i in range(len(matrix)):
        d={}
        counter=0
        for j in range(len(matrix[i])):
            d[cls[j]]=matrix[i][j]
            counter+=matrix[i][j]
        d['classError']=((counter-matrix[i][i])/counter)
        arr.append(d)
    print(arr[0])
    # np.savetxt("matC.txt",np.array(arr),delimiter=",")
    file.write(str(arr))







def cnfu_matrix(Model,xt,yt,d,name):
    y_pred = Model.predict(xt)
    cnf_matrix = confusion_matrix(yt, y_pred)
    np.set_printoptions(precision=2)
    class_names = []
    for key, value in d.items():
        temp = key
        class_names.append(temp)
    np.array(class_names)
    exportConfusion(class_names,cnf_matrix)
    plt.figure(figsize=(20, 20))
    plot_confusion_matrix(name=name,cm=cnf_matrix, classes=class_names,title='Confusion matrix, without normalization ')

    plt.show()



def plot_confusion_matrix(name,cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        np.save(name,cm)
        print("Normalized confusion matrix")
    else:


        print('Confusion matrix, without normalization')


    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_size=[10,10]
    plt.rcParams["figure.figsize"] = fig_size


def load_data_from_csv(sizeOfDataSet=0,file=None):
    # pass 0 in sizeOfDataSet to get full dataset
    file=open(file,'r')
    data=[]
    cls=[]
    index=-1
    for video in file:
        if(sizeOfDataSet!=0):
            if(index ==sizeOfDataSet):
                break;
        if index==-1:
            index+=1
            continue;
        features=video.split(',')
        data.append([])
        for i in range(1,581):
            data[index].append(int(features[i]))
            if(i==580):
                cls.append(features[581][:-1])
                # cls[len(cls)-1]=cls[len(cls)-1][:-1]
        index+=1
    return np.array(data,dtype=float),np.array(cls)

def load_data_from_csv_word(sizeOfDataSet=0, file=None):
    # pass 0 in sizeOfDataSet to get full dataset
    file = open(file, 'r')
    data = []
    dict={}
    cls = []
    index = -1
    for video in file:
        if (sizeOfDataSet != 0):
            if (index == sizeOfDataSet):
                break;

        if index == -1:
            index += 1
            continue;

        features = video.split(',')
        if features[581] not in dict:
            dict[features[581]]=1

        # print(features[581] != 'BANKS')
        data.append([])
        # print(1)
        for i in range(1, 581):
            data[index].append(int(features[i]))
            if (i == 580):
                cls.append(features[581][:-1])
                # cls[len(cls)-1]=cls[len(cls)-1][:-1]
        index += 1
    print('why')
    return np.array(data), np.array(cls),dict






def load_data_from_csv_word_inDiction(sizeOfDataSet=0, file=None,dict=None):
    # pass 0 in sizeOfDataSet to get full dataset
    file = open(file, 'r')
    data = []

    cls = []
    index = -1
    for video in file:
        if (sizeOfDataSet != 0):
            if (index == sizeOfDataSet):
                break;

        # if index == -1:
        #     index += 1
        #     continue;

        features = video.split(',')
        if features[291] not in dict:

            continue;

        # print(features[581] != 'BANKS')
        data.append([])
        # print(1)
        for i in range(1, 291):
            data[index].append(float(features[i]))
            if (i == 290):
                cls.append(features[291][:-1])
                # cls[len(cls)-1]=cls[len(cls)-1][:-1]
        index += 1

    return np.array(data), np.array(cls)






def load_data_from_csv_wordCount(sizeOfDataSet=0, file=None):
    # pass 0 in sizeOfDataSet to get full dataset
    print(file)
    file = open(file, 'r')
    data = []
    dict={}
    newDict={}
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
        # print(len(features))
        # if(len(features)<580) or len(features)> 590:
        #     continue;
        # if features[1101]=='28.30194339616981' :
        #     print("why")
        if features[291][:-1] not in dict:
            dict[features[291][:-1]]=1
            newDict[features[291][:-1]]=[]
        else  :
            dict[features[291][:-1]]+=1
        # elif dict[features[581]] == 950:
        #     continue;
        # # print(features[581] != 'BANKS')
        data.append([])

        # print(1)
        array_features=[]
        for i in range(1, 291):
            data[index].append(float(features[i]))
            if (i == 290):
                cls.append(features[291][:-1])

        # yield (data, cls)
        index += 1
    # print('why')
    if sizeOfDataSet !=0:
        dict.popitem()
        newDict.popitem()
        data=data[:len(data)-1]
        cls = cls[:len(cls) - 1]
    # print(index)
    # data.extend(data)
    # cls.extend(cls)
    return np.array(data,dtype=float), np.array(cls) ,dict
















x, y, d = load_data_from_csv_wordCount(sizeOfDataSet=5, file="Geometric.csv")
xt, xtest, yt, ytest = train_test_split(x, y, test_size=0.2, random_state=2)
model=GaussianNB()
model.fit(xt,yt)
# cnfu_matrix(model,xtest,ytest,d,"Naive")
print(model.score(xtest,ytest))





# filter = "JSON file (*.json)|*.json|All Files (*.*)|*.*||"
# filename ="/home/octazero/Desktop/Dataset/"
#
#
#
# #
# cls=[]
# dataset=[]
# arr=[]
# fi = []
# for (dirpath, dirnames, filenames) in walk(filename):
#     fi.extend(filenames)
# fi.sort()
# for file in fi:
#     FILE=open(filename+file,'r')
#     print(filename + file)
#     str=FILE.read()
#     FILE.close()
#     str=str.replace("}{","},{")
#     str='{"array":['+str
#     str=str+']}'
#     FILE=open(filename+file,'w')
#     FILE.write(str)
#     FILE.close()


# for file in fi:
#     print(file)
#
#     if filename+file:
#         with open(filename+file, 'r') as f:
#             datastore = json.load(f)
#         for data in datastore["array"]:
#             for array in data['sequence']:
#                 for element in array:
#                     arr.append(element)
#             if len(arr)==580:
#                 cls.append(data['class'])
#                 dataset.append(arr)
#             arr=[]
# #
# print(len(dataset))
# #
# # #Read JSON data into the datastore variable
# file = "333word.csv"
# wr = open(file, 'w')
# wr.write(",0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,label\n")
# for i  in range(len(dataset)):
#     wri=''
#     for wordd in dataset[i]:
#         wri+=str(wordd)+','
#     wr.write(str(i+1)+','+wri+cls[i]+'\n')
# wr.close()

# print(len(dataset[0]))
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# # Model=svm.SVC()
# # Model.fit(np.array(dataset),np.array(cls))
# # joblib.dump(Model, '23word.pkl')
# # print(Min," ",Sec)
# # print(datetime.datetime.now().minute," ",datetime.datetime.now().second)
# #
# mod=joblib.load('23word.pkl')
# print(Min," ",Sec)
# print(datetime.datetime.now().minute," ",datetime.datetime.now().second)
# # print(len(trst))
# for val in dataset:
#     print(mod.predict(np.array([val])))

#




#poly100
# day=datetime.datetime.now().day
# hr=Min = datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print("start sigmoid")
# Model=svm.SVC(kernel='poly',C=0.25,gamma=100.0000000000000001e-09,cache_size=1000)
# print(Model.get_params())
# x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=100,file="data_269_word.csv")
# print(len(d))
# xt,yt=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="test_dataset.csv",dict=d)
# print(len(x))
# print(len(y))
# print("fitting")
# print(day," ",hr," ",Min," ",Sec)
# # scores = cross_val_score(Model, x,y, cv=5)
# # Model.fit(x,y)
# # joblib.dump(Model, '100poly.pkl')
# # print("done",scores.mean())
#
# print("done")
# # print(Model.score(xt,yt))
# Model=joblib.load('100poly.pkl')
# y_pred=Model.predict(xt)
#
#
# day=datetime.datetime.now().day
# hr=Min = datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print(day," ",hr," ",Min," ",Sec)
# cnf_matrix = confusion_matrix(yt, y_pred)
# np.set_printoptions(precision=2)
# plt.figure(figsize=(100,100))
# class_names=[]
#
# for key, value in d.items():
#     temp = key
#     class_names.append(temp)
# np.array(class_names)
# file=open("classes",'w')
# np.save("classes.bin", class_names)
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization ')
#
#
#
# plt.figure(figsize=(100,100))
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
# plt.show()








# x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=10,file="Geometric.csv")
# x=x.astype('float')/x.max()
# x,xtest,y,ytest=train_test_split(x, y, test_size=0.2, random_state=2)
# xtest=xtest.reshape(len(xtest),29,10)
#naive
# xtest[0][0]=np.delete(xtest[0][0],1)
#
# from keras.utils import to_categorical
# uniques, id_test=np.unique(ytest,return_inverse=True)
# ytest=to_categorical(id_test,10)
#
# import conf
# from keras.models import load_model
# Model=load_model("saved10/weights.11.h5")
# Model.predict([xtest[0]])
# # conf.cnfu_matrix(uniques,Model,xtest,ytest,"LSTM")
# # print(counter)





#
# x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=100,file="data_269_word.csv")
# print(len(d))
# print(len(x))
# xt,yt=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="test_dataset.csv",dict=d)
#
# # Model=svm.SVC(kernel='poly',C=0.01,gamma=1.0000000000000001e-09,decision_function_shape='ovo')
# # day=datetime.datetime.now().day
# # hr=Min = datetime.datetime.now().hour
# # Min = datetime.datetime.now().minute
# # Sec = datetime.datetime.now().second
# #
# # print("fitting")
# # print(day," ",hr," ",Min," ",Sec)
# #
# # Model.fit((x/200),y)
# # day=datetime.datetime.now().day
# # hr=Min = datetime.datetime.now().hour
# # Min = datetime.datetime.now().minute
# # Sec = datetime.datetime.now().second
# # print(day," ",hr," ",Min," ",Sec)
# # joblib.dump(Model, '23naive.pkl')
# Model=joblib.load("100poly.pkl")
# print(Model.score(xt,yt))
# cnfu_matrix(Model,xt,yt,d)
#
#






    #randomForest
# day=datetime.datetime.now().day
# hr=Min = datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# x,y,d=load_data_from_csv_word(sizeOfDataSet=0,file="data_269_word.csv")
# print(len(d))
# xt,yt=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="test_dataset.csv",dict=d)
# print(len(xt))
# m=RandomForestClassifier(max_depth=100)
# m.fit(x,y)
# print(day," ",hr," ",Min," ",Sec)
# day=datetime.datetime.now().day
# hr=Min = datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print(day," ",hr," ",Min," ",Sec)
# print(m.score(xt,yt))


def load_dataset():
    x, y, d = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/ASKED.csv")
    x1, y1, d1 = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/AMOUNT.csv")
    x = np.concatenate((x, x1))
    y = np.concatenate((y, y1))
    d = {**d, **d1}
    x1, y1, d1 = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/ABOUT.csv")
    x = np.concatenate((x, x1))
    y = np.concatenate((y, y1))
    d = {**d, **d1}
    x1, y1, d1 = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/BENEFIT.csv")
    x = np.concatenate((x, x1))
    y = np.concatenate((y, y1))
    d = {**d, **d1}
    x1, y1, d1 = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/BEFORE.csv")
    x = np.concatenate((x, x1))
    y = np.concatenate((y, y1))
    d = {**d, **d1}
    return x,y


#5 word Confusiony
def fun():
    print("why was as")
    x, y, d = load_data_from_csv_wordCount(sizeOfDataSet=5, file="Geometric.csv")
    MaxScore = 0.0
    index = 0

    xt, xtest, yt, ytest = train_test_split(x, y, test_size=0.2, random_state=2)
    # xt=x[:-3]
    # xtest=x[-3:]
    # yt = y[:-3]
    # ytest = y[-3:]
    # print(len(xtest))
    # print(len(xt))
    # print("why was as")
    # x,y=load_dataset()
    # print(len(y))
    # xtest,ytest=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="test_dataset.csv",dict=d)
    # xt,yt=load_data_from_csv_word_inDiction(sizeOfDataSet=0,file="validate_dataset.csv",dict=d)
    #120

    # for i in range(0,1000):
    # gamma_range = np.logspace(-9, 3, 13)
    # for i in gamma_range:
    # Model=GradientBoostingClassifier(n_estimators=40000)
    Model = svm.SVC(kernel='poly', C=0.01, gamma=1.0000000000000001e-05,probability=True)

    Model.fit(xt,yt)
    score = Model.score(xtest, ytest)

    print(score)
    # print(index)
    # joblib.dump(Model, 'SVMCustomDataset.pkl')
    # print(Model.predict([xtest[0]]))
    # print(Model.predict_proba([xtest[0]]))
        # print(i)
    # cnfu_matrix(Model,xtest,ytest,d=d,name="SVM")

        # if(MaxScore<score):
        #     index=i
        #     MaxScore=score
        #     joblib.dump(Model, 'highest_SVM.pkl')
        #     print(str(score) + "----------------------------------------------- ", i)
    #     print(str(score)+" ",i)
    # print(index)
    # print(Model.predict([xtest[200]]), " ", ytest[150])
# fun()
# x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=5,file="Geometric.csv")
# Model=LogisticRegression(C=0.0001,random_state=0)
# xt,xtest,yt,ytest=train_test_split(x, y, test_size=0.2, random_state=77)
# Model.fit(xt,yt)
# score = Model.score(xtest, ytest)
# print(score)
# print(Model.predict([xtest[200]])," ",ytest[150])

def extraTree():
    print("hello")
    x, y, d = load_data_from_csv_wordCount(sizeOfDataSet=0, file="ourInterpolate.csv")
    # xtest, ytest = load_data_from_csv_word_inDiction(sizeOfDataSet=0, file="test_dataset.csv", dict=d)
    xt, xtest, yt, ytest = train_test_split(x, y, test_size=0.2, random_state=2)
    Model=ExtraTreesClassifier(n_estimators=400)
    Model.fit(xt,yt)
    print(Model.score(xtest,ytest))


    # joblib.dump(Model, 'ExtraTree.pkl')
    # cnfu_matrix(Model, xtest, ytest, d=d, name="ExtraTree")

    Model=GradientBoostingClassifier(n_estimators=400)
    Model.fit(xt, yt)
    print(Model.score(xtest, ytest))
    # joblib.dump(Model, 'GradientBoost.pkl')
    # cnfu_matrix(Model, xtest, ytest, d=d, name="GradientBoosting")

# extraTree()



# x,y,d=load_data_from_csv_word(0,'data.csv')
#
# x,y=load_data_from_csv_word_inDiction(0,'test_dataset.csv',d)
# print(len(x))
#
#
#
# Model=svm.SVC()
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# # print(len(x))
# Model=joblib.load('23poly.pkl')
# print(Model.score(x,y))
# print(Min," ",Sec)
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print(Min," ",Sec)


#grid Search


# # x,y=load_data_from_csv_word_inDiction(0,'test_dataset.csv',d)
# x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=0,file="/home/octazero/Downloads/geoData/ASKED.csv")
# print(len(y))
# x1, y1, d = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/AMOUNT.csv")
# x=np.concatenate((x,x1))
# y=np.concatenate((y,y1))
# print(len(y1))
# x1, y1, d = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/ABOUT.csv")
# x = np.concatenate((x, x1))
# y = np.concatenate((y, y1))
# print(len(y1))
# x1, y1, d = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/BENEFIT.csv")
# x = np.concatenate((x, x1))
# y = np.concatenate((y, y1))
# print(len(y1))
# x1, y1, d = load_data_from_csv_wordCount(sizeOfDataSet=0, file="/home/octazero/Downloads/geoData/BEFORE.csv")
# x = np.concatenate((x, x1))
# y = np.concatenate((y, y1))
# print(len(y1))
# # print(y)
# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# print("resutl")
# grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)
# da=datetime.datetime.now().day
# hr=datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print(da," ",hr," ",Min," ",Sec)
#
# grid.fit(x, y)
#
# da=datetime.datetime.now().day
# hr=datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print(da," ",hr," ",Min," ",Sec)
#
#
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
# joblib.dump(grid, 'gridsearchGeoElGded.pkl')


# grid=joblib.load('geometric5Words.pkl')
#
# print(grid.best_params_)




# day=datetime.datetime.now().day
# hr=Min = datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print("start linear")
# Model=svm.SVC(kernel='linear')
# print(Model)
# # x,y=load_data_from_csv(sizeOfDataSet=0,file="videos.csv")
# print("fitting")
# print(day," ",hr," ",Min," ",Sec)
# scores = cross_val_score(Model, x, y, cv=5)
#
# print("done ",scores.mean())
# print(day," ",hr," ",Min," ",Sec)
# day=datetime.datetime.now().day
# hr=Min = datetime.datetime.now().hour
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# print(day," ",hr," ",Min," ",Sec)





# Model.fit(x,y)
# print("saving")
#
# print("done")
# print(Min," ",Sec)
# print(datetime.datetime.now().minute," ",datetime.datetime.now().second)
# print("dumping")
# Min = datetime.datetime.now().minute
# Sec = datetime.datetime.now().second
# joblib.dump(Model, '23wordLinearKernel.pkl')
#
# print(Min," ",Sec)
# print(datetime.datetime.now().minute," ",datetime.datetime.now().second)

# for array in data:
#     cls=data[-1]
#     data=data[1:]
# dataset,cls=load_data_from_csv(sizeOfDataSet=0,file='test_dataset.csv')
# print(len(dataset))
# mod=joblib.load('23word.pkl')
#
# result=mod.predict(dataset)
# print(result)
# counter=0
# for i in range(len(result)):
#     result=mod.predict(np.array([dataset[i]]))
#     if result==cls[i]:
#         print(result)
#         counter+=1
# print(counter)
#
# model = Sequential()
# model.add(Embedding(128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train...')
# model.fit(np.array(datastore["array"][0]['sequence']), y_train,
#           batch_size=batch_size,
#           epochs=15,
#           validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)




#
# data=np.array([[]])
# for i in range(10):
#     data=np.append(data,0)
# features=np.array([])
# print(data)
# data=np.expand_dims(data,axis=0)
# for i in range(10):
#     for j in range(10):
#         features=np.append(features,j)
#     # print(features)
#     data=np.append(data,[features],axis=0)
#     features=np.array([])
# data=np.delete(data,0,0)
# #
# # # data[0]=np.array([])
# # # data[0]=np.append(data,[5])
# # data=np.append(data,[[10,5]])
# print (data)
#
#




# file =open("500words.txt",'r')
# array=[]
# for l in file:
#     l=l[:-1]
#     array.append(l)
#
# np.save('wo',array)
# array=np.array(array)
# print(np.where(array=="BETWEEN")[0][0])
# print(array)


#Voting Algorithm
#
#
# print("hello")
#
# def plot_confusion_matri(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         # print("Normalized confusion matrix")
#
#
#     # print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# #
# #
#
# SVMMatrix = np.load("SVM.npy")
# CNNMatrix = np.load("CNN.npy")
# LSTMMatrix = np.load("LSTM.npy")
# GradientBoostMatrix = np.load("GradientBoosting.npy")
# ExtraTreeMatrix = np.load("ExtraTree.npy")
# Matricies= {"SVM": SVMMatrix, "ExtraTree": ExtraTreeMatrix, "Gradient": GradientBoostMatrix,
#                      "LSTM": LSTMMatrix, "CNN": CNNMatrix}
# #
# # #models
# ExtraTreeModel = joblib.load('ExtraTree.pkl')
# SVMModel = joblib.load('SVM.pkl')
# CNNModel = load_model("CNN.h5")
# LSTMModel = load_model("LSTM.h5")
# GradientBoostModel = joblib.load('GradientBoost.pkl')
#
# ListOfClassifiers = {
#                       "CNN": CNNModel,"LSTM":LSTMModel}
#
# Words=np.load("words.npy")
# classifierAccuracy={"CNN":0.72,"LSTM":0.683,"SVM":0.61,"ExtraTree":0.618,"Gradient":0.63}
# # print(ListOfClassifiers)
# def predict(x):
#     Result={}
#     #confusionMatrix
#     predictedWords = [0,0,0,0,0,0,0,0,0,0]
#     for k,v in ListOfClassifiers.items():
#         if k == "LSTM" or k == "CNN":
#             f=x
#             f=np.reshape(x,(29,10))
#             Result[k]=v.predict(np.array([f]))
#         else:
#             Result[k] = v.predict_proba([x])
#
#         for prediction in range(len(Result[k][0])):
#             predictedWords[prediction]=predictedWords[prediction]+\
#                                        ((classifierAccuracy[k]*Result[k][0,prediction]))
#     # print(predictedWords)
#     # print(Words[np.argmax(predictedWords)]," ",np.argmax(predictedWords))
#     # Sumprobalilties=0
#     # for i in range(len(predictedWords)):
#     #     Sumprobalilties+=predictedWords[i]
#     # if predictedWords[np.argmax(predictedWords)]/Sumprobalilties <0.5:
#     #     print(predictedWords[np.argmax(predictedWords)]/Sumprobalilties)
#     #     return "false"
#     return Words[np.argmax(predictedWords)]
#
#
# def score(x,y):
#     ypred = []
#     counter=0
#     for i in range(len(x)):
#         # day=datetime.datetime.now().day
#         # hr=Min = datetime.datetime.now().hour
#         # Min = datetime.datetime.now().minute
#         # Sec = datetime.datetime.now().second
#         # print(day," ",hr," ",Min," ",Sec)
#         result=predict(x[i])
#         if result=="false":
#             counter+=1
#
#         # print(day," ",hr," ",Min," ",Sec)
#         ypred.append(result)
#
#     print(counter)
#     cnf_matrix = confusion_matrix(y, ypred)
#     np.set_printoptions(precision=2)
#
#     # Plot non-normalized confusion matrix
#
#     # Plot normalized confusion matrix
#     plt.figure()
#     plot_confusion_matri(cnf_matrix, classes=Words, normalize=True,
#                           title='Normalized confusion matrix')
#
#     plt.show()
#     return accuracy_score(y,ypred)
#
#
# x,y,d=load_data_from_csv_wordCount(sizeOfDataSet=10,file="Geometric.csv")
# x,xtest,y,ytest=train_test_split(x, y, test_size=0.2, random_state=2)
#
#
#
# # r=predict([25.0, 72.44998274671983, 77.89736837660179, 94.92101980067429, 138.6542462386205, 2.0, 111.01801655587259, 70.22819946431775, 75.17978451685, 98.65596788841515, 18.0, 78.10889834071403, 77.89736837660179, 94.92101980067429, 136.56500283747664, 0.0, 120.30793822520607, 80.62257748298549, 80.99382692526635, 94.92101980067429, 25.0, 84.77027781009096, 72.71863585079137, 94.72592042308166, 139.5313584826006, 0.0, 123.664869708418, 84.77027781009096, 72.71863585079137, 94.72592042308166, 18.0, 83.81527307120105, 77.41446893184762, 94.92101980067429, 134.23859355639868, 4.0, 117.72000679578642, 81.27115109311545, 74.43117626371358, 97.86725703727473, 25.96150997149434, 72.44998274671983, 69.31810730249349, 93.7709976485267, 136.1653406708183, 2.0, 116.29703349613007, 64.62197768561404, 68.41052550594829, 90.79647570252934, 24.73863375370596, 64.62197768561404, 68.41052550594829, 90.79647570252934, 136.12494260788506, 7.211102550927978, 117.6860229593982, 61.326992425847855, 67.67569726275453, 93.7709976485267, 18.0, 59.09314681077663, 68.41052550594829, 97.18538984847466, 130.17296186228538, 6.0, 113.63538181394033, 59.09314681077663, 68.41052550594829, 97.18538984847466, 17.0, 63.694583757176716, 67.00746227100382, 94.19129471453293, 133.5402561027947, 0.0, 110.07270324653611, 65.29931086925804, 71.02816342831905, 94.64142856064674, 23.0, 74.0, 68.60029154456998, 94.76286192385707, 136.85393673548452, 6.4031242374328485, 119.20151005754919, 75.8946638440411, 69.9714227381436, 97.73944955850733, 24.73863375370596, 64.62197768561404, 68.41052550594829, 97.18538984847466, 132.5028301584536, 8.48528137423857, 111.01801655587259, 64.62197768561404, 68.41052550594829, 97.18538984847466, 17.0, 60.90155991434045, 62.369864518050704, 100.17983829094555, 128.03515142334936, 7.810249675906654, 112.71202242884297, 63.694583757176716, 75.43208866258443, 100.71742649611338, 17.0, 81.70679286326198, 80.2122185206219, 102.83968105745953, 127.63228431709588, 1.0, 114.93476410555685, 85.09406559801923, 86.40023148117139, 105.53672346628922, 17.0, 84.15461959987698, 85.60373823613078, 106.7379969832674, 133.2816566523691, 5.0990195135927845, 115.6027681329474, 91.21403400793103, 90.21086409075129, 104.4030650891055, 17.0, 92.97311439335567, 85.14693182963201, 104.12012293500234, 133.5402561027947, 6.0, 118.06777714516353, 87.31551981177229, 85.14693182963201, 102.83968105745953, 17.0, 74.0, 76.8505042273634, 101.6070863670443, 136.85393673548452, 4.123105625617661, 119.20151005754919, 72.49827584156743, 78.23042886243178, 98.65596788841515, 18.0, 70.22819946431775, 81.60882305241266, 98.65596788841515, 133.33041663476493, 2.0, 120.7683733433551, 66.85057965343307, 67.67569726275453, 93.90420650854784, 18.0, 66.85057965343307, 76.02631123499285, 100.84145972763385, 128.94960255851896, 4.123105625617661, 112.71202242884297, 66.18912297349165, 71.84705978674423, 100.71742649611338, 18.0, 59.09314681077663, 68.41052550594829, 97.73944955850733, 132.5028301584536, 6.0, 111.01801655587259, 67.08203932499369, 73.23933369440222, 103.69667304209909, 19.0, 69.07966415668217, 78.2368199762746, 97.86725703727473, 135.08515832614626, 6.0, 118.20744477400736, 69.07966415668217, 73.40980860893181, 97.18538984847466, 25.0, 61.326992425847855, 71.21797525905943, 93.1933474020544, 137.80058055030102, 4.0, 113.01769772916099, 63.56099432828282, 73.40980860893181, 97.18538984847466, 20.0, 65.79513659838392, 72.11102550927978, 100.24470060806208, 133.71985641631537, 7.0, 116.1077086157504, 71.8679344353238, 70.83784299369935, 100.0, 20.0, 72.2357252334328, 72.11102550927978, 100.24470060806208, 130.3878828726044, 4.0, 114.93476410555685, 63.56099432828282, 69.8927750200262, 97.86725703727473, 20.0, 72.2357252334328, 66.40030120413611, 93.0, 133.71985641631537, 7.0, 116.1077086157504, 69.92138442565336, 66.40030120413611, 100.9752444909147, 19.0, 63.56099432828282, 68.76772498781678, 97.86725703727473, 134.23859355639868, 6.324555320336759, 109.41663493271945, 59.135437767890075, 70.60453243241541, 100.17983829094555, 19.0, 63.56099432828282, 73.40980860893181, 97.25224933131366, 126.7162183779172, 8.48528137423857, 114.93476410555685, 69.07966415668217, 73.40980860893181, 97.18538984847466, 19.0, 63.56099432828282, 73.40980860893181, 97.86725703727473, 132.23085872821065, 6.0, 114.93476410555685, 63.56099432828282, 69.33974329343886, 97.18538984847466, 25.0, 61.326992425847855, 67.67569726275453, 93.1933474020544, 137.38267721950973, 2.0, 117.72000679578642, 63.56099432828282, 65.0, 97.0, 19.0, 63.56099432828282, 73.40980860893181, 97.25224933131366, 128.80993750483694, 10.0, 120.7683733433551, 61.326992425847855, 71.21797525905943, 93.1933474020544, 19.0, 63.56099432828282, 73.40980860893181, 97.25224933131366, 137.7679207943562, 6.0, 118.20744477400736, 63.56099432828282, 73.40980860893181, 97.18538984847466])
# print(score(xtest,ytest))
# # predict(r)