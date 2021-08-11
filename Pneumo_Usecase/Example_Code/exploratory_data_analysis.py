import os
import os.path as osp
import glob
from multiprocessing.spawn import freeze_support

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader

# from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


data_dir = './chest_xray/'

###########################################################################################
## Function for plotting data distribution
###########################################################################################
# train_pneumonia_dir = osp.join(data_dir, 'train/PNEUMONIA')
# train_pneumonia_img_list = glob.glob(train_pneumonia_dir + '/*.jpeg')
# print("Number of training samples of pneumonia", len(train_pneumonia_img_list))
# train_normal_dir = osp.join(data_dir, 'train/NORMAL')
# train_normal_img_list = glob.glob(train_normal_dir + '/*.jpeg')
# print("Number of training samples of normal", len(train_normal_img_list))
#
# train_samplesize = pd.DataFrame.from_dict(
#     {'Normal': [len([os.path.join(data_dir + '/train/NORMAL', filename)
#                      for filename in os.listdir(data_dir + '/train/NORMAL')])],
#      'Pneumonia': [len([os.path.join(data_dir + '/train/PNEUMONIA', filename)
#                         for filename in os.listdir(data_dir + '/train/PNEUMONIA')])]})
# print(train_samplesize)
# plt.figure()
# sns.barplot(data=train_samplesize)
# plt.title('Training Set Data Inbalance', fontsize=20)
# test_samplesize = pd.DataFrame.from_dict(
#     {'Normal': [len([os.path.join(data_dir+'/test/NORMAL', filename)
#                      for filename in os.listdir(data_dir+'/test/NORMAL')])],
#      'Pneumonia': [len([os.path.join(data_dir+'/test/PNEUMONIA', filename)
#                         for filename in os.listdir(data_dir+'/test/PNEUMONIA')])]})
# plt.figure()
# sns.barplot(data=test_samplesize).set_title('Test Set Data Inbalance', fontsize=20)
# plt.show()


###########################################################################################
## Function for plotting samples
###########################################################################################
# def plot_samples(samples):
#     fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 8))
#     for i in range(len(samples)):
#         image = cv2.cvtColor(imread(samples[i]), cv2.COLOR_BGR2RGB)
#         ax[i//5][i%5].imshow(image)
#         if i<5:
#             ax[i//5][i%5].set_title("Normal", fontsize=20)
#         else:
#             ax[i//5][i%5].set_title("Pneumonia", fontsize=20)
#         ax[i//5][i%5].axis('off')
# rand_samples = random.sample([os.path.join(data_dir + '/train/NORMAL', filename)
#                               for filename in os.listdir(data_dir + '/train/NORMAL')], 5) + \
#                random.sample([os.path.join(data_dir + '/train/PNEUMONIA', filename)
#                               for filename in os.listdir(data_dir + '/train/PNEUMONIA')], 5)
# plt.figure()
# plot_samples(rand_samples)
# plt.suptitle('Training Set Samples', fontsize=30)


###########################################################################################
## Define training dataset
###########################################################################################
dataset = ImageFolder(data_dir + '/train',
                      transform=tt.Compose([tt.Resize(255),
                                            tt.CenterCrop(224),
                                            tt.RandomHorizontalFlip(),
                                            tt.RandomRotation(10),
                                            tt.RandomGrayscale(),
                                            tt.RandomAffine(translate=(0.05, 0.05), degrees=0),
                                            tt.ToTensor() #, tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ,inplace=True)
                                            ]))

# print(train_dataset.classes)
# img, label = train_dataset[0]
# print(img.shape, label)
train_size = round(len(dataset)*0.7) # 70%
val_size = len(dataset) - train_size # 30%
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

###########################################################################################
## Function for plotting sample
###########################################################################################
# def show_example(img, label):
#     print('Label: ', dataset.classes[label], "(" + str(label) + ")")
#     plt.imshow(img.permute(1, 2, 0))
# plt.figure()
# show_example(*train_ds[4])


batch_size=128
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2, shuffle=True)
###########################################################################################
## Function for plotting sample
###########################################################################################
# def show_batch(dl):
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(12, 12))
#         ax.set_xticks([]);
#         ax.set_yticks([])
#         ax.imshow(make_grid(images[:60], nrow=10).permute(1, 2, 0))
#         break
# plt.figure()
# show_batch(train_dl)
# plt.show()

torch.cuda.is_available()
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()






























# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
# import os
# import shutil
# from collections import defaultdict
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.layers import Conv2D,MaxPool2D
# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, confusion_matrix
# from joblib.numpy_pickle_utils import xrange
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.layers import Conv2D,MaxPool2D
#
# def define_model(img_size):
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
#     model.add(MaxPool2D(2, 2))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPool2D(2, 2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def process_data(img_dims, batch_size, stationNumber):
#     # Data generation objects
#     train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, vertical_flip=True)
#     # This is fed to the network in the specified batch sizes and image dimensions
#     train_gen = train_datagen.flow_from_directory(
#         directory=station_path + str(stationNumber),
#         target_size=(img_dims, img_dims),
#         batch_size=batch_size,
#         class_mode='binary',
#         shuffle=True)
#     return train_gen
#
# def get_test_data(img_dims,batch_size):
#     test_val_datagen = ImageDataGenerator(rescale=1. / 255)
#     test_gen = test_val_datagen.flow_from_directory(
#         directory=input_path + 'test',
#         target_size=(img_dims, img_dims),
#         batch_size=batch_size,
#         class_mode='binary',
#         shuffle=True)
#     # I will be making predictions off of the test set in one batch size
#     # This is useful to be able to get the confusion matrix
#     test_data = []
#     test_labels = []
#     for cond in ['/NORMAL/', '/PNEUMONIA/']:
#         for img in (os.listdir(input_path + 'test' + cond)):
#             img = plt.imread(input_path + 'test' + cond + img)
#             img = cv2.resize(img, (img_dims, img_dims))
#             img = np.dstack([img, img, img])
#             img = img.astype('float32') / 255
#             if cond == '/NORMAL/':
#                 label = 0
#             elif cond == '/PNEUMONIA/':
#                 label = 1
#             test_data.append(img)
#             test_labels.append(label)
#     test_data = np.array(test_data)
#     test_labels = np.array(test_labels)
#     return test_gen, test_data, test_labels
#
# def splitInChunks(a, n):
#     k, m = divmod(len(a), n)
#     return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))
#
# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_dims, img_dims, 3)))
#     model.add(MaxPool2D(2, 2))
#     model.add(Conv2D(32, (3, 3), activation='relu'))
#     model.add(MaxPool2D(2, 2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model
#
# #################################################################################
# # Hyperparameters
# img_dims = 100
# epochs = 1
# batch_size = 1
# institutionHops = 9
# # Setting seeds for reproducibility
# #seed = 232
# #tf.random.set_seed(seed)
# for numberStations in [3]:
#     stationArray = range(1, numberStations+1)
#     #################################################################################
#     #################################################################################
#     ################################Distribute Data##################################
#     #################################################################################
#     #################################################################################
#     train = []
#     labels = []
#     input_path = 'chest_xray/'
#     output_path = 'EqualDistribution/StationFolders/'
#     station_path = output_path + str(numberStations) + "Stations/"
#     for reps in range(1):
#         if os.path.exists(output_path + "/" + str(numberStations) + "Stations" + "/") and os.path.isdir(output_path + "/" + str(numberStations) + "Stations" + "/"):
#             shutil.rmtree(output_path + "/" + str(numberStations) + "Stations" + "/")
#         for station in range(1,numberStations+1):
#             os.makedirs(output_path + "/" + str(numberStations) + "Stations" + "/" + str(station) + '/NORMAL/')
#             os.makedirs(output_path + "/" + str(numberStations) + "Stations" + "/" + str(station) + '/PNEUMONIA/')
#         train = []
#         labels = []
#         for cond in ['/NORMAL/', '/PNEUMONIA/']:
#             for img in (os.listdir(input_path + 'train' + cond)):
#                 if cond == '/NORMAL/':
#                     label = 0
#                 elif cond == '/PNEUMONIA/':
#                     label = 1
#                 train.append(img)
#                 labels.append(label)
#         train, labels = shuffle(train, labels)
#         fileNameChunks = list(splitInChunks(train, numberStations))
#         fileLabelChunks = list(splitInChunks(labels, numberStations))
#         experimentPath = output_path + "/" + str(numberStations) + "Stations"
#         for station in range(1,numberStations+1):
#             stationPath = experimentPath + "/" + str(station)
#             currentStationFileNameChunk = fileNameChunks[station-1]
#             currentStationFileLabelChunk = fileLabelChunks[station-1]
#             for file, label in zip(currentStationFileNameChunk, currentStationFileLabelChunk):
#                 if label == 0:
#                     labelDir = '/NORMAL/'
#                 elif label == 1:
#                     labelDir = '/PNEUMONIA/'
#                 shutil.copy2(input_path + 'train' + labelDir + "/" + file, stationPath + labelDir)
#         #################################################################################
#         #################################################################################
#         dataGens = []
#         for station in range(1,numberStations+1):
#             dataGens.append(process_data(img_dims,batch_size,station))
#         # Getting the test data
#         test_gen, test_data, test_labels = get_test_data(img_dims,batch_size)
#         perm = [(1,2,3)]
#         print(len(perm))
#         permutationDict = defaultdict(list)
#         for permutation in perm:
#             model = tf.keras.models.load_model('initModel')
#             accHistory = []
#             preds = model.predict(test_data)
#             acc = accuracy_score(test_labels, np.round(preds)) * 100
#             accHistory.append(acc)
#             # Fitting the model
#             for iteration in range(institutionHops):
#                 print("+++++++++++++++++++Iteration "+str(iteration+1)+"++++++++++++++++++++++")
#                 print("+++++++++++++++++++Station  "+str(permutation[iteration%numberStations])+"++++++++++++++++++++++")
#                 #loss = model.evaluate_generator(dataGens[station%numberStations])
#                 #acc = model.predict_generator(dataGens[station%numberStations], steps=dataGens[station%numberStations].samples // batch_size)
#                 history = model.fit_generator(
#                    dataGens[permutation[iteration%numberStations]-1], steps_per_epoch=dataGens[permutation[iteration%numberStations]-1].samples,
#                    epochs=epochs, validation_data=test_gen, shuffle=True,
#                    validation_steps=test_gen.samples // batch_size)
#                 preds = model.predict(test_data)
#                 acc = accuracy_score(test_labels, np.round(preds)) * 100
#                 accHistory.append(acc)
#             permutationDict[permutation].append(accHistory)
#             preds = model.predict(test_data)
#             acc = accuracy_score(test_labels, np.round(preds))*100
#             cm = confusion_matrix(test_labels, np.round(preds))
#             tn, fp, fn, tp = cm.ravel()
#             print('CONFUSION MATRIX ------------------')
#             print(cm)
#             print('\nTEST METRICS ----------------------')
#             precision = tp/(tp+fp)*100
#             recall = tp/(tp+fn)*100
#             print('Accuracy: {}%'.format(acc))
#             print('Precision: {}%'.format(precision))
#             print('Recall: {}%'.format(recall))
#             print('F1-score: {}'.format(2*precision*recall/(precision+recall)))
#             print(accHistory)
#         print(permutationDict)
