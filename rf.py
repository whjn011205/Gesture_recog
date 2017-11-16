import os,sys
import cv2
import numpy as np
import collections
import random
from numba import jit
import cProfile
from sklearn.ensemble import RandomForestClassifier
from feature import getFeatures,crop,patchDetection
import datetime

now=datetime.datetime.now

category={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'A':11,'B':12,'C':13,'D':14,'E':15,'F':3,'G':17,'H':18,'I':19,'K':20,'L':8,'M':21,'N':22,'O':23,'P':24,'Q':25,'R':26,'S':10,'T':27,'U':28,'V':2,'W':29,'wave':5,'X':9,'Y':6,'Z':30}
u=np.array(random.sample(range(1,20000),20)) #generate 20 random numbers for the x and y coordinate of 10 u's
v=np.array(random.sample(range(1,20000),20)) #generate 20 random numbers for the x and y coordinate of 10 v's


print("\n------------------Generating training data--------------")
t0=now()

path = 'C://Users//WH//Desktop//SSF-filtered//SSF-PNG//'
subjectFolders = os.listdir(path)
subjectFolders = [folder for folder in subjectFolders if 'ssf' in folder]

first_train=True
first_test=True
for subjectFolder in subjectFolders[0:1]: # get the subject folders
    
    gestures = os.listdir(path+subjectFolder)
    for gesture in gestures: # get the list of gestures in this subject
        #print(gesture)
        files = os.listdir(path+subjectFolder+'//'+gesture)
        handtxt = [img for img in files if "txt" in img] # get the txt files for this gesture
        imgNames1 = [str.split(title,'.')[0] for title in handtxt]

        handjpg = [img for img in files if "jpg" in img] # get the jpg files for this gesture
        imgNames2 = [str.split(title,'.')[0] for title in handjpg]
        
        imgNames = list(set(imgNames1) & set(imgNames2)) # an image file must have both jpg and txt
        for title in imgNames:
            #print(title)
            try:
                tempX = getFeatures(path,subjectFolder, title,gesture,u,v)
                if tempX is None:
                    continue
                    
                tempX=tempX.T
                tempY=np.zeros((len(tempX),1),dtype='int16') + category[gesture]
                
                if int(title[4:])!=1: #if odd res, such as res_1, res_3
                    #print("odd")
                    if first_train:
                        X_train=tempX
                        Y_train=tempY[:,0]
                        first_train=False
                    else:
                        X_train=np.concatenate((X_train,tempX),axis=0)
                        Y_train=np.concatenate((Y_train,tempY[:,0]),axis=0)
                
                
            except Exception:
                pass
                #print ("bad image: "+subjectFolder+'//'+gesture+'//'+title)

np.savetxt("X_train.csv", X_train, delimiter=",")  
np.savetxt("Y_train.csv", Y_train, delimiter=",")           
print(X_train.shape,Y_train.shape)

t1=now()
print(t1-t0)

print("\n------------------Training---------------------")
t0=now()
clf=RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(X_train,Y_train)
t1=now()
print(t1-t0)


print("\n----------------Predicting---------------------") 
result_list=[]
t0=now()    

first_train=True
first_test=True
for subjectFolder in subjectFolders[0:1]: # get the subject folders
    gestures = os.listdir(path+subjectFolder)
    for gesture in gestures: # get the list of gestures in this subject
        #print(gesture)
        files = os.listdir(path+subjectFolder+'//'+gesture)
        handtxt = [img for img in files if "txt" in img] # get the txt files for this gesture
        imgNames1 = [str.split(title,'.')[0] for title in handtxt]

        handjpg = [img for img in files if "jpg" in img] # get the jpg files for this gesture
        imgNames2 = [str.split(title,'.')[0] for title in handjpg]
        
        imgNames = list(set(imgNames1) & set(imgNames2)) # an image file must have both jpg and txt
        for title in imgNames:
            #print(title)
            try:
                tempX = getFeatures(path,subjectFolder, title,gesture,u,v)
                if tempX is None:
                    continue
                
                tempX=tempX.T
                tempY=np.zeros((len(tempX),1),dtype='int16') + category[gesture]
                
                if int(title[4:])==1: #if odd res, such as res_1, res_3
                    if first_test:
                        X_test=tempX
                        Y_test=tempY[:,0]
                        first_test=False
                    else:
                        X_test=np.concatenate((X_test,tempX),axis=0)
                        Y_test=np.concatenate((Y_test,tempY[:,0]),axis=0)
            except Exception:
                pass
                #print ("bad image: "+subjectFolder+'//'+gesture+'//'+title)

    print(X_test.shape, Y_test.shape)
    Y_pred = clf.predict(X_test)
    
    print('\n')
    print('sub '+subjectFolder[7:].split('-')[0]+':')
    
    a= sum((1 for i in range(len(Y_pred)) if Y_pred[i] == Y_test[i]))
    print ("\taccuracy1: %.2f"%(a/len(Y_test)))
    
    for i in range(int(len(Y_pred)/10)):
        result_list.append((Y_test[10*i],np.bincount(Y_pred[10*i:10*(i+1)]).argmax()))
    
    i=0
    for test,pred in result_list:
        if test==pred:
            i+=1
    print("\taccuracy2: %.2f" %(i/len(result_list)) )
    

    d=collections.defaultdict(int)
    for item in result_list:
        if item[0]!=item[1]:
            d[item]+=1
    i=0        
    for key in sorted(d, key=d.get, reverse=True):
        if i>3: 
            break
        print("\tmost error rate:",key,d[key])
        i+=1
    
            
t1=now()
print("\n")
print(t1-t0)
print("\n\n")

