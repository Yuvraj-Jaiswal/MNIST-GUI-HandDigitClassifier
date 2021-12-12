from tensorflow.python.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers \
    import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

single_image = x_train[0]
#plt.imshow(single_image , cmap='binary')

y_train_mat = to_categorical(y_train , 10)
y_test_mat = to_categorical(y_test , 10)

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

early_stop = EarlyStopping(monitor='val_loss' , patience=1)

model = Sequential()
model.add(Conv2D(filters=32 , kernel_size=(4,4)
                 , input_shape=(28,28,1) , activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=16 , kernel_size=(4,4)
                 , input_shape=(28,28,1) , activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128 , activation='relu'))
model.add(Dense(10 , activation='softmax'))

model.compile(loss='categorical_crossentropy' , optimizer='adam'
              ,metrics=['accuracy'])
model.fit(x_train , y_train_mat , epochs=50 ,
          validation_data=(x_test,y_test_mat) , callbacks=[early_stop])

#%%
metric_loss = pd.DataFrame(model.history.history)
metric_loss[['loss' , 'val_loss']].plot()
metric_loss[['accuracy' , 'val_accuracy']].plot()
#%%
from sklearn.metrics import confusion_matrix

prediction = model.predict_classes(x_test)

confusion_matrix(y_test,prediction)

#%%
from sklearn.metrics import classification_report
classification_report(y_test,prediction)
#%%
model.predict_classes(x_test[5].reshape(1,28,28,1))

#%%
model.save('DMM.h5')

