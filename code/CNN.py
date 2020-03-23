import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras import regularizers
from keras.constraints import maxnorm
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

num_data_train = 40000
num_data_test = 6800
test_cut = 37500
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

features_train = np.hstack((features_train,np.square(features_train)))
#,np.power(features_train,3),np.power(features_train,4)))
features_test = np.hstack((features_test,np.square(features_test)))
#,np.power(features_test,3),np.power(features_test,4)))
print(features_train.shape)
print(features_test.shape)

#Defining labels according to the correspondance
labels_train = np.full((num_data_train,1),1)
labels_test =  np.full((num_data_test,1),1)
for i in range(2,num_classes+1):
	labels_train = np.vstack((labels_train,np.full((num_data_train,1),i)))
	labels_test = np.vstack((labels_test,np.full((num_data_test,1),i)))
	
print(labels_train.shape)
print(labels_test.shape)

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

#one-hot encoding
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)
labels_train = labels_train[:,1:]
labels_test = labels_test[:,1:]

#Declaring type for any kind of normalization
features_train = features_train.astype("float32")
features_test = features_test.astype("float32")

#Reshaping to form a 4-D tensor
features_train = np.reshape(features_train,(num_classes*num_data_train,28,28,2))
features_test = np.reshape(features_test,(num_classes*num_data_test,28,28,2))


model = keras.Sequential([
		tf.keras.layers.Conv2D(128,(3,3),input_shape = (28,28,2),padding = 'same',activation = 'sigmoid',kernel_regularizer=keras.regularizers.l2(0.005)),
		tf.keras.layers.MaxPool2D(pool_size = 2),
		tf.keras.layers.Dropout(0.3),
		
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128,activation = 'sigmoid',),
		tf.keras.layers.Dense(num_classes, activation = 'softmax')
		])

epoch = 30
model.summary()
model.compile(optimizer = keras.optimizers.SGD(lr = 0.01),loss='categorical_crossentropy', metrics=['accuracy'])		
acc = model.fit(features_train,labels_train,validation_split = 0.2,epochs = epoch, batch_size = 32)

#Plot the accuracy curves
axis = []
for i in range(1,epoch+1):
	axis.append(i)
plt.plot(axis,acc.history['acc'])
plt.plot(axis,acc.history['val_acc'])
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.show()

#saving the model to local directory
json = model.to_json()
with open("../model/model2.json","w") as json_file:
	json_file.write(json)
model.save_weights("../model/model2.h5")

#Final testing accuracy
test_loss, test_acc = model.evaluate(features_test,labels_test)
print('Test accuracy:', test_acc)




