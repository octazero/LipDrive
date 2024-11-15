import numpy as np
import matplotlib.pyplot as plt
import itertools
# from sklearn.metrics import confusion_matrix

def cnfu_matrix(classnames,Model,xt,yt,name):

    matrix=[]
    for i in range(len(classnames)):
        matrix.append([])
        for j in range(len(classnames)):
            matrix[i].append(0)
    for i in range(len(yt)):
        result=Model.predict_proba(np.array([xt[i]]))
        matrix[np.argmax(yt[i])][np.argmax(result)]+=1
        # print(yt[i])
    # cnf_matrix = confusion_matrix(yt, y_pred)
    matrix=np.array(matrix)
    np.set_printoptions(precision=2)
    # class_names = []
    # for key, value in d.items():
    #     temp = key
    #     class_names.append(temp)
    np.array(classnames)
    plt.figure(figsize=(20, 20))
    plot_confusion_matrix(name=name,cm=matrix, classes=classnames,title='Confusion matrix, without normalization ')

    plt.show()


def plot_confusion_matrix(name, cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        np.save(name, cm)
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
    import operator
    fmt = '.2f' if normalize else 'd'
    thresh = np.amax(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_size = [10, 10]
    plt.rcParams["figure.figsize"] = fig_size