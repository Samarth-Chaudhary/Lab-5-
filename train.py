import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

train_dir='dataset/train'
test_dir='dataset/test'

train_gen=ImageDataGenerator(rescale=1./255,rotation_range=20,zoom_range=0.2,horizontal_flip=True)
test_gen=ImageDataGenerator(rescale=1./255)

train=train_gen.flow_from_directory(train_dir,target_size=(128,128),batch_size=32,class_mode='binary')
test=test_gen.flow_from_directory(test_dir,target_size=(128,128),batch_size=32,class_mode='binary')

model=Sequential([
Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
MaxPooling2D(2,2),
Conv2D(64,(3,3),activation='relu'),
MaxPooling2D(2,2),
Conv2D(128,(3,3),activation='relu'),
MaxPooling2D(2,2),
Flatten(),
Dense(128,activation='relu'),
Dropout(0.5),
Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train,epochs=10,validation_data=test)
loss,acc=model.evaluate(test)
print('Test Accuracy:',acc)
model.save('fresh_rotten_classifier.h5')
