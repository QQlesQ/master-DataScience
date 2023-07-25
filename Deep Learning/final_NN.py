import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Dropout
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from tqdm.contrib import tzip
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def augmentation(paths, label_data):
    aug_img = []
    aug_labels = []
    for path, data in zip(paths, label_data):
        images = data['id'].tolist()
        label = [[rows.label1, rows.label2, rows.label3] for rows in data.itertuples()]   
        
        for i, j in tzip(images, label):
            img  = cv2.imread(os.path.join(path, i))
            aug_img.append(tf.keras.utils.img_to_array(img)/255.0)
            aug_img.append(tf.keras.utils.img_to_array(np.fliplr(img))/255.0)
            aug_img.append(tf.keras.utils.img_to_array(np.flipud(img))/255.0)
#             aug_img.append(tf.keras.utils.img_to_array(rotate(img, angle=-45))/255.0)
#             aug_img.append(tf.keras.utils.img_to_array(rotate(img, angle=+45))/255.0)
            aug_img.append(tf.keras.utils.img_to_array(adjust_gamma(img, gamma=0.5, gain=1))/255.0)
            aug_img.append(tf.keras.utils.img_to_array(adjust_gamma(img, gamma=2, gain=1))/255.0)
#             aug_img.append(tf.keras.utils.img_to_array(rotate(img, angle=-135))/255.0)
#             aug_img.append(tf.keras.utils.img_to_array(rotate(img, angle=+135))/255.0)

            aug_labels += ([j] * 5)
    return np.array(aug_img), np.array(aug_labels)


def plot_accuracy_loss(history):
    fig = plt.figure(figsize=(10,5))
    
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label='acc')
    plt.plot(history.history['val_accuracy'],'ro--', label='val_acc')
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label='loss')
    plt.plot(history.history['val_loss'],'ro--', label='val_loss')
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    
    plt.legend()
    plt.show()


random_state = 42

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


train_dir = "train_images"
vali_dir = "val_images"


train_data = pd.read_csv('train_labels.csv')
vali_data = pd.read_csv('val_labels.csv')


aug_img, aug_labels = augmentation([train_dir ,vali_dir], [train_data, vali_data])

x_train, x_vali, y_train, y_vali = train_test_split(aug_img, aug_labels, test_size=0.1, random_state=random_state, shuffle = True)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_vali", x_vali.shape)
print("y_vali", y_vali.shape)

# NN

model = Sequential()

model.add(Flatten(input_shape=(64,64,3)))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='sigmoid'))

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                         restore_best_weights=True)
model.summary()

hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_vali, y_vali), callbacks=callback)

plot_accuracy_loss(hist)

predictions = model.predict(x_vali)
pred = np.matrix.round(predictions).astype(int)

label = ['label1', 'label2', 'label3']

print(classification_report(y_vali, pred, target_names=label))

test_data = pd.read_csv('sample_submission.csv')
test_dir = "test_images"

test = []
for i in tqdm(range(test_data.shape[0])):
    path = f'{test_dir}/'+test_data['id'][i]
    img = tf.keras.utils.load_img(path,target_size=(64,64,3))
    img = tf.keras.utils.img_to_array(img)
    img = img/255.0
    test.append(img)

test = np.array(test) 

predictions = model.predict(test)
predictions = np.matrix.round(predictions).astype(int)
predictions = pd.DataFrame(predictions,columns=["label1", 'label2', 'label3'])
sub = pd.concat([test_data.id,predictions],axis=1)
sub.set_index('id',inplace=True)
sub.to_csv(f"Submission_x.csv")
