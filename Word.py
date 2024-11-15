import numpy as np
class Word:
    def __init__(self,name,sequences):
        self.sequences=sequences
        self.name=name
    def Train(self):
        training=[]
        name=[]
        for seq in self.sequences:
            training.append(seq)
            name.append(self.name)

        return np.array(training),np.array(name)