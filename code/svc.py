import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
from sklearn.svm import SVC


num_data_train = 4700
num_data_test = 1715
test_cut = 24567
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

#Taking some data rows from each class as the features to be trained
features_train = np.vstack((data1[0:num_data_train,:],data2[0:num_data_train,:],data3[0:num_data_train,:],
							data4[0:num_data_train,:],data5[0:num_data_train,:],data6[0:num_data_train,:],
							data7[0:num_data_train,:],data8[0:num_data_train,:],data9[0:num_data_train,:],
							data10[0:num_data_train,:],data11[0:num_data_train,:],data12[0:num_data_train,:],
							data13[0:num_data_train,:],data14[0:num_data_train,:],data15[0:num_data_train,:],
							data16[0:num_data_train,:],data17[0:num_data_train,:],data18[0:num_data_train,:],
							data19[0:num_data_train,:],data20[0:num_data_train,:]))

#Taking some data rows from each class as the features to be tested
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

#Declaring type for any kind of normalization
features_train = features_train.astype("float32")
features_test = features_test.astype("float32")
features_train = features_train/255.0
features_test = features_test/255.0

#Defining labels according to the correspondance
labels_train = np.full((num_data_train,1),1)
labels_test =  np.full((num_data_test,1),1)
for i in range(2,num_classes+1):
	labels_train = np.vstack((labels_train,np.full((num_data_train,1),i)))
	labels_test = np.vstack((labels_test,np.full((num_data_test,1),i)))

#Combining the labels and features for a shuffle
data_train = np.hstack((features_train,labels_train))
data_test = np.hstack((features_test,labels_test))
print(data_train.shape)
print(data_test.shape)
np.random.shuffle(data_train)
np.random.shuffle(data_test)

#Separating the shuffled array
features_train = data_train[:,:-1]
features_test = data_test[:,:-1]
labels_train = data_train[:,features_train.shape[1]]
labels_test = data_test[:,features_train.shape[1]]

val = [1.3,1.7,2.0,2.7,3.0,3.5,5.0]
training_accuracy = []
val_accuracy = []
opt_val = 0
opt_acc = 0

t = time.time()
classy = SVC(verbose=True,gamma=0.02,C = 3.5)
classy.fit(features_train,labels_train)
dats = open('../model/svc23.pkl','wb') 
pickle.dump(classy, dats)
dats.close()
acc = classy.score(features_test,labels_test)
print("Training Accuracy:" ,classy.score(features_train,labels_train))
print("Validation Accuracy:", acc)
print("Time Taken: ",time.time()-t)

#More data to train on
#Use Nystroem
#Read 
#Add confusion matrix to all 
#KNN and logreg for comparison, possibly naive bayes. Make sparse documentation. 