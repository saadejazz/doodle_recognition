import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


#Stats
num_data_test = 100
test_cut = random.randint(55500,75000)
num_classes = 20

#Defining a correspondance between label value and label names
labels = ['axe','clock','mountain','skull','triangle','lion','fish','airplane','cloud','bat',
			'cup','apple','star','octagon','camel','umbrella','leaf','duck','diamond','house']

#Loading dataset for 20 different classes
data1 = np.load('../dataset/axe.npy')
data2 = np.load('../dataset/clock.npy')
data3 = np.load('../dataset/mountain.npy')
data4 = np.load('../dataset/skull.npy')
data5 = np.load('../dataset/triangle.npy')
data6 = np.load('../dataset/lion.npy')
data7 = np.load('../dataset/fish.npy')
data8 = np.load('../dataset/airplane.npy')
data9 = np.load('../dataset/cloud.npy')
data10 = np.load('../dataset/bat.npy')
data11 = np.load('../dataset/cup.npy')
data12 = np.load('../dataset/apple.npy')
data13 = np.load('../dataset/star.npy')
data14 = np.load('../dataset/octagon.npy')
data15 = np.load('../dataset/camel.npy')
data16 = np.load('../dataset/umbrella.npy')
data17 = np.load('../dataset/leaf.npy')
data18 = np.load('../dataset/duck.npy')
data19 = np.load('../dataset/diamond.npy')
data20 = np.load('../dataset/house.npy')

#Taking portions for testing
features_test = np.vstack((data1[test_cut:test_cut+num_data_test,:],data2[test_cut:test_cut+num_data_test,:],
							data3[test_cut:test_cut+num_data_test,:],data4[test_cut:test_cut+num_data_test,:],
							data5[test_cut:test_cut+num_data_test,:],data6[test_cut:test_cut+num_data_test,:],
							data7[test_cut:test_cut+num_data_test,:],data8[test_cut:test_cut+num_data_test,:],
							data9[test_cut:test_cut+num_data_test,:],data10[test_cut:test_cut+num_data_test,:],
							data11[test_cut:test_cut+num_data_test,:],data12[test_cut:test_cut+num_data_test,:],
							data13[test_cut:test_cut+num_data_test,:],data14[test_cut:test_cut+num_data_test,:],
							data15[test_cut:test_cut+num_data_test,:],data16[test_cut:test_cut+num_data_test,:],
							data17[test_cut:test_cut+num_data_test,:],data18[test_cut:test_cut+num_data_test,:],
							data19[test_cut:test_cut+num_data_test,:],data20[test_cut:test_cut+num_data_test,:]))

#Adding squared features							
features_test = np.hstack((features_test,np.square(features_test)))
labels_test =  np.full((num_data_test,1),1)
for i in range(2,num_classes+1):
	labels_test = np.vstack((labels_test,np.full((num_data_test,1),i)))
	
#Combining the labels and features for a shuffle
data_test = np.hstack((features_test,labels_test))
np.random.shuffle(data_test)

#Separating the shuffled array
features_test = data_test[:,:-1]
labels_test = data_test[:,features_test.shape[1]]

#one-hot encoding
labels_test = to_categorical(labels_test)
labels_test = labels_test[:,1:]

#Declaring type for any kind of normalization
features_test = features_test.astype("float32")
features_test = features_test/255.0

#Reshaping to form a 4-D tensor
features_test = np.reshape(features_test,(num_classes*num_data_test,28,28,2))

#Loading saved model							
json_file = open("../model/logreg.json")
json = json_file.read()
json_file.close()
model = model_from_json(json)
model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])
model.load_weights("../model/logreg.h5")

#Testing
print("Running Algorithm on test dataset")
test_loss, test_acc = model.evaluate(features_test,labels_test)
print('Test accuracy:', test_acc)
y_true = labels_test.argmax(axis = 1)
y_pred = model.predict(features_test).argmax(axis = 1)
matrix = confusion_matrix(y_true,y_pred)
print(matrix)