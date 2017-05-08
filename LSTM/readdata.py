import csv
import numpy as np



class Data:
    def __init__(self):
        self.features=[]
        self.label=[]




def readdata(featurepath,labelpath):
    labels = {}
    with open(labelpath) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            label_=np.array(np.zeros(157))
            label=row[9].split(';')
            for i in range(0,len(label)):
                label[i]=int(label[i][1:4])
                label_[label[i]]=1
            labels.update({row[0]:label_})#eg.  46GP8: [0,0,0,0,...1,....,0]
	
    with open(featurepath) as ff:
        ff_csv = csv.reader(ff)
        headers = next(f_csv)
        for row in ff_csv:   #return data with .features and .label

    data = Data()
    data.label=
    data.features=
