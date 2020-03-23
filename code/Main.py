import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from keras.utils import to_categorical
import pickle

# num_data_train = 2000
# num_data_test = 30000
# test_cut = random.randint(40000,75000)
# num_classes = 20

# #Defining a correspondance between label value and label names
# labels = ['axe','clock','mountain','skull','triangle','lion','fish','airplane','cloud','bat',
# 			'cup','apple','star','octagon','camel','umbrella','leaf','duck','diamond','house']

# #Loading dataset for 20 different classes
# data1 = np.load('../dataset/axe.npy')
# data2 = np.load('../dataset/clock.npy')
# data3 = np.load('../dataset/mountain.npy')
# data4 = np.load('../dataset/skull.npy')
# data5 = np.load('../dataset/triangle.npy')
# data6 = np.load('../dataset/lion.npy')
# data7 = np.load('../dataset/fish.npy')
# data8 = np.load('../dataset/airplane.npy')
# data9 = np.load('../dataset/cloud.npy')
# data10 = np.load('../dataset/bat.npy')
# data11 = np.load('../dataset/cup.npy')
# data12 = np.load('../dataset/apple.npy')
# data13 = np.load('../dataset/star.npy')
# data14 = np.load('../dataset/octagon.npy')
# data15 = np.load('../dataset/camel.npy')
# data16 = np.load('../dataset/umbrella.npy')
# data17 = np.load('../dataset/leaf.npy')
# data18 = np.load('../dataset/duck.npy')
# data19 = np.load('../dataset/diamond.npy')
# data20 = np.load('../dataset/house.npy')

# #Taking some data rows from each class as the features to be trained
# features_train = np.vstack((data1[0:num_data_train,:],data2[0:num_data_train,:],data3[0:num_data_train,:],
# 							data4[0:num_data_train,:],data5[0:num_data_train,:],data6[0:num_data_train,:],
# 							data7[0:num_data_train,:],data8[0:num_data_train,:],data9[0:num_data_train,:],
# 							data10[0:num_data_train,:],data11[0:num_data_train,:],data12[0:num_data_train,:],
# 							data13[0:num_data_train,:],data14[0:num_data_train,:],data15[0:num_data_train,:],
# 							data16[0:num_data_train,:],data17[0:num_data_train,:],data18[0:num_data_train,:],
# 							data19[0:num_data_train,:],data20[0:num_data_train,:]))

# #Taking some data rows from each class as the features to be tested
# features_test = np.vstack((data1[test_cut:test_cut+num_data_test,:],data2[test_cut:test_cut+num_data_test,:],
# 							data3[test_cut:test_cut+num_data_test,:],data4[test_cut:test_cut+num_data_test,:],
# 							data5[test_cut:test_cut+num_data_test,:],data6[test_cut:test_cut+num_data_test,:],
# 							data7[test_cut:test_cut+num_data_test,:],data8[test_cut:test_cut+num_data_test,:],
# 							data9[test_cut:test_cut+num_data_test,:],data10[test_cut:test_cut+num_data_test,:],
# 							data11[test_cut:test_cut+num_data_test,:],data12[test_cut:test_cut+num_data_test,:],
# 							data13[test_cut:test_cut+num_data_test,:],data14[test_cut:test_cut+num_data_test,:],
# 							data15[test_cut:test_cut+num_data_test,:],data16[test_cut:test_cut+num_data_test,:],
# 							data17[test_cut:test_cut+num_data_test,:],data18[test_cut:test_cut+num_data_test,:],
# 							data19[test_cut:test_cut+num_data_test,:],data20[test_cut:test_cut+num_data_test,:]))


# #Defining labels according to the correspondance
# labels_train = np.full((num_data_train,1),1)
# labels_test =  np.full((num_data_test,1),1)
# for i in range(2,num_classes+1):
# 	labels_train = np.vstack((labels_train,np.full((num_data_train,1),i)))
# 	labels_test = np.vstack((labels_test,np.full((num_data_test,1),i)))
	
# #Combining the labels and features for a shuffle
# data_train = np.hstack((features_train,labels_train))
# data_test = np.hstack((features_test,labels_test))
# np.random.shuffle(data_train)
# np.random.shuffle(data_test)

# #Separating the shuffled array
# features_train = data_train[:,:-1]
# features_test = data_test[:,:-1]
# labels_train = data_train[:,features_train.shape[1]]
# labels_test = data_test[:,features_train.shape[1]]

# #Declaring type for any kind of normalization
# features_train = features_train.astype("float32")
# features_test = features_test.astype("float32")

# #KNN classifier
# KNN = KNeighborsClassifier(n_neighbors = 6)
# KNN.fit(features_train,labels_train)

#Loading CNN model
json_file = open("../model/CNN.json")
json = json_file.read()
json_file.close()
modely = model_from_json(json)
modely.compile(loss = 'categorical_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])
modely.load_weights("../model/CNN.h5")

#Loading LogReg model
json_file = open("../model/logreg.json")
json = json_file.read()
json_file.close()
model = model_from_json(json)
model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])
model.load_weights("../model/logreg.h5")

#Loading SVC model
foo = open('../model/svc.pkl','rb')
classy = pickle.load(foo)
foo.close()

#numerical preprocessing for image showing
j = random.randint(0,num_data_test*num_classes-1)
datax = features_test[j]
datax = np.reshape(datax,(28,28))

#show image on a plot
plt.figure()
plt.imshow(datax)
plt.colorbar()
plt.grid(False)
plt.show()

#preprocessing for prediction
datax = np.reshape(datax,(1,784))
datax = datax.astype('float32')
datax = datax/255.0
datak = datax

#predicting through SVM
ans = classy.predict(datax)
ans = ans.astype('int32')
k = ans[0]-1
print('The true label is: ',labels[labels_test[j]-1])
print('SVM predicts ',labels[k])

#Predicting through KNN
datax = datax * 255.0
ans = KNN.predict(datax)
ans = ans.astype('int32')
k = ans[0]-1
print('KNN predicts ',labels[k])

#Predicting through CNN
datax = np.hstack((datax,np.power(datax,2)))
datax = np.reshape(datax, (datax.shape[0],28,28,2))
ans = modely.predict(datax)
result = []
prediction = np.argmax(ans,axis = 1)
for i in prediction:
	result.append(labels[i])
print("CNN predicts ",result[0])

#Predicting through LogRegS
datak = np.hstack((datak,np.power(datak,2)))
datak = np.reshape(datak, (datak.shape[0],28,28,2))
ans = model.predict(datak)
prediction = np.argmax(ans,axis = 1)
for i in prediction:
	result[0] = labels[i]
print("LogReg predicts ",result[0])
