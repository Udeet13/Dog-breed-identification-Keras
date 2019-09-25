import warnings
warnings.filterwarnings('ignore')

import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join

train_folder = 'train/'
test_folder = 'train/'

train_dogs = pd.read_csv('labels.csv')
train_dogs.head()

#graph specifying images per class
ax = pd.value_counts(train_dogs['breed'],ascending=True).plot(kind='barh',fontsize="40",title="Image per class",figsize=(50,100))
ax.set(xlabel="Images", ylabel="classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()

top_breeds = sorted(list(train_dogs['breed'].value_counts().head(50).index))
train_dogs=train_dogs[train_dogs['breed'].isin(top_breeds)]

print(top_breeds)
train_dogs.shape
target_labels = train_dogs['breed']

one_hot=pd.get_dummies(target_labels, sparse = True)
one_hot_labels = np.asarray(one_hot)

train_dogs['image_path'] = train_dogs.apply( lambda x:(train_folder + x["id"] + ".jpg"), axis=1)
train_dogs.head()

#convert images to arrays which will be use in model.
train_data = np.array([img_to_array(load_img(img, target_size=(299,299))) for img in train_dogs['image_path'].values.tolist()])

#split the data into train and validation.

x_train, x_validation, y_train, y_validation = train_test_split(train_data, target_labels, test_size=0.2, stratify = np.array(target_labels), random_state=100)

print('x_train shape = ', x_train.shape)
print('x_validation shape = ', x_validation.shape)

#graph representation 
data = y_train.value_counts().sort_index().to_frame()

data.columns = ['train']

data['validation']= y_validation.value_counts().sort_index().to_frame()

new_plot = data[['train','validation']].sort_values(['train']+['validation'], ascending=False)

new_plot.plot(kind='bar', stacked=True)

plt.show()

#converting the train and validation labels into one hot format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_validation=pd.get_dummies(y_validation.reset_index(drop=True)).as_matrix()

#train generator
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip='true')

train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=10, seed=10)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=10,seed=10)

base_model = InceptionV3(weights = 'imagenet', include_top=False, input_shape=(299,299,3))

#Global spatial average pooling layer
x= base_model.output
x=GlobalAveragePooling2D()(x)

#fully-connected layer and a logistsic layer with 50 classes
#(there will be 120 classes for the final submission)
x=Dense(512, activation='relu')(x)
predictions = Dense(50,activation="softmax")(x) 

#the model will train
model = Model(input = base_model.input, outputs = predictions)

#first: only training top layers
for layer in base_model.layers:
    layer.trainable = False
    
#compile with Adam
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit_generator(train_generator,steps_per_epoch=175,validation_data=val_generator,validation_steps=44,epochs=2,verbose=1)
model.save("retrained.h5")


from keras.applications.inception_v3 import preprocess_input
import cv2
image=cv2.imread("Cairn1.jpg")
arrayresized = cv2.resize(image,(299,299))
inputarray = arrayresized[np.newaxis,...]
im = preprocess_input(inputarray)
predict=model.predict(im)
y_classes = predict.argmax(axis=-1)
y_classes=max(y_classes)
print(top_breeds[y_classes])
