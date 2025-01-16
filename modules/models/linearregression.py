import numpy as np
import os
import random

class LR:
    def __init__(self,input):
        self.input = input
        self.initparams()

    def initparams(self):
        self.weights = np.random.rand(self.input,1).astype(float)
        self.bais = float(random.random())

    def Calculategrad(self,x,y):
        n = len(x)
        y_pred = np.dot(x,self.weights)+self.bais
        loss = self.calError(y_pred,y)
        dw = (2/n)*np.dot(x.T,(y_pred.squeeze()-y))
        db = (2/n)*np.sum(y_pred.squeeze()-y)
        return loss,dw,db

    def Train(self,epochs,data,y):
        for i in range(epochs):
            loss,dw,db = self.Calculategrad(data,y)
            print("Epoch Loss:",loss)
            self.weights -= 0.001*dw.reshape(1,1)
            self.bais -= 0.001*db

    def pred(self,data):
        return np.dot(data,self.weights) + self.bais
    
    def calError(self,pred,org):
        return np.mean((pred - org)**2)


if __name__ == "__main__":

    final_dataset = None

    with open("../../datasets/basic.csv","r") as f:
        line = f.read()
        data = []
        lines = line.split('\n')
        data = [single.split(',') for single in lines]
        final_dataset = np.array(data)
    testdata = [i for i in range(30)]
    testdatay = [1000*testdata[i] for i in range(30)]
    ans = np.array(final_dataset[:,1]).astype(float)
    new_data = np.array(final_dataset[:,0],dtype=float).reshape(len(final_dataset[:,0]),1)
    model = LR(1)
    ouput = model.Train(100000,np.array(testdata).reshape(30,1),testdatay)
    