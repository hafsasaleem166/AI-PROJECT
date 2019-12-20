import pandas as pd
import numpy as np
datafile=pd.read_csv("C:/Users/Hafsa/Desktop/datafile.csv")
inputs=np.array(datafile[['x1','x2','x3']])
inputs.tolist()
labels=datafile["y"]
labels.tolist()
#Initilization of Hyperparameters
alpha=0.1
threshold=1
iterations=100
n = 0
#Function to update the weights
def Learning(inputs,labels,threshold,iterations,alpha):
    w = np.zeros(len(inputs[0]))
    n=0
    while n<iterations:
        for i in range(0, len(inputs)):
             predicted_output = np.dot(inputs[i], w)
             # activation function
             if predicted_output > threshold:
                output = 1.
             else:
                output= 0.
             for j in range (0,len(w)):
                 error=labels[i]-output
                 w[j]=w[j]+alpha*(error)*inputs[i][j] #updated weights formula
             file=open("C:/Users/Hafsa/Desktop/test1.txt","w")
             for f in w:
                file.write("%s\n" %f)
             file.close()
             print("Given inputs ",inputs[i])
             print("Wgeights updated",w)
             print("Errors in iterations",error)
        n=n+1


    return w

def prediction():
    #Testing of the weights which is given in test1.txt file
    w=[]
    weights=open("C:/Users/Hafsa/Desktop/test1.txt","r")
    weights=[x.rstrip("\n") for x in weights.readlines()]
    print(weights)
    for i in range(0,len(inputs)-1):
        w.append(float(weights[i]))

    for i in range(0, len(inputs)):
        predicted_output = np.dot(inputs[i], w)
        # activation function
        if predicted_output > threshold:
            output= 1.
        else:
            output= 0.
        print("predicted output",output)



if __name__== "__main__" :
    w=Learning(inputs,labels,threshold,iterations,alpha)
    prediction()