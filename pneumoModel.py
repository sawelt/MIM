
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import shutil
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D,MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib.numpy_pickle_utils import xrange






def process_data(img_dims, batch_size, stationNumber):
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, vertical_flip=True)

    # This is fed to the network in the specified batch sizes and image dimensions
    train_gen = train_datagen.flow_from_directory(
        directory=station_path + str(stationNumber),
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)



    return train_gen




def get_test_data(img_dims,batch_size):

    test_val_datagen = ImageDataGenerator(rescale=1. / 255)

    test_gen = test_val_datagen.flow_from_directory(
        directory=input_path + 'test',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data = []
    test_labels = []

    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path + 'test' + cond + img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond == '/NORMAL/':
                label = 0
            elif cond == '/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return test_gen, test_data, test_labels

def splitInChunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_dims, img_dims, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



#################################################################################
# Hyperparameters
img_dims = 100
epochs = 1
batch_size = 1
institutionHops = 9

# Setting seeds for reproducibility
#seed = 232
#tf.random.set_seed(seed)



for numberStations in [3]:


    stationArray = range(1, numberStations+1)

    #################################################################################
    #################################################################################
    ################################Distribute Data##################################
    #################################################################################
    #################################################################################


    train = []
    labels = []


    input_path = 'chest_xray/'
    output_path = 'EqualDistribution/StationFolders/'
    station_path = output_path + str(numberStations) + "Stations/"


    for reps in range(1):


        if os.path.exists(output_path + "/" + str(numberStations) + "Stations" + "/") and os.path.isdir(output_path + "/" + str(numberStations) + "Stations" + "/"):
            shutil.rmtree(output_path + "/" + str(numberStations) + "Stations" + "/")


        for station in range(1,numberStations+1):

            os.makedirs(output_path + "/" + str(numberStations) + "Stations" + "/" + str(station) + '/NORMAL/')
            os.makedirs(output_path + "/" + str(numberStations) + "Stations" + "/" + str(station) + '/PNEUMONIA/')

        train = []
        labels = []
        for cond in ['/NORMAL/', '/PNEUMONIA/']:

            for img in (os.listdir(input_path + 'train' + cond)):


                if cond == '/NORMAL/':
                    label = 0
                elif cond == '/PNEUMONIA/':
                    label = 1
                train.append(img)
                labels.append(label)



        train, labels = shuffle(train, labels)

        fileNameChunks = list(splitInChunks(train, numberStations))
        fileLabelChunks = list(splitInChunks(labels, numberStations))



        experimentPath = output_path + "/" + str(numberStations) + "Stations"


        for station in range(1,numberStations+1):
            stationPath = experimentPath + "/" + str(station)

            currentStationFileNameChunk = fileNameChunks[station-1]
            currentStationFileLabelChunk = fileLabelChunks[station-1]

            for file, label in zip(currentStationFileNameChunk, currentStationFileLabelChunk):

                if label == 0:
                    labelDir = '/NORMAL/'
                elif label == 1:
                    labelDir = '/PNEUMONIA/'

                shutil.copy2(input_path + 'train' + labelDir + "/" + file, stationPath + labelDir)




        #################################################################################
        #################################################################################





        dataGens = []
        for station in range(1,numberStations+1):
            dataGens.append(process_data(img_dims,batch_size,station))



        # Getting the test data
        test_gen, test_data, test_labels = get_test_data(img_dims,batch_size)





        perm = [(1,2,3)]
        print(len(perm))
        permutationDict = defaultdict(list)



        for permutation in perm:


            model = tf.keras.models.load_model('initModel')


            accHistory = []
            preds = model.predict(test_data)
            acc = accuracy_score(test_labels, np.round(preds)) * 100
            accHistory.append(acc)

            # Fitting the model
            for iteration in range(institutionHops):
                print("+++++++++++++++++++Iteration "+str(iteration+1)+"++++++++++++++++++++++")
                print("+++++++++++++++++++Station  "+str(permutation[iteration%numberStations])+"++++++++++++++++++++++")
                #loss = model.evaluate_generator(dataGens[station%numberStations])
                #acc = model.predict_generator(dataGens[station%numberStations], steps=dataGens[station%numberStations].samples // batch_size)
                history = model.fit_generator(
                   dataGens[permutation[iteration%numberStations]-1], steps_per_epoch=dataGens[permutation[iteration%numberStations]-1].samples,
                   epochs=epochs, validation_data=test_gen, shuffle=True,
                   validation_steps=test_gen.samples // batch_size)

                preds = model.predict(test_data)
                acc = accuracy_score(test_labels, np.round(preds)) * 100
                accHistory.append(acc)
            permutationDict[permutation].append(accHistory)


            preds = model.predict(test_data)


            acc = accuracy_score(test_labels, np.round(preds))*100
            cm = confusion_matrix(test_labels, np.round(preds))
            tn, fp, fn, tp = cm.ravel()

            print('CONFUSION MATRIX ------------------')
            print(cm)

            print('\nTEST METRICS ----------------------')
            precision = tp/(tp+fp)*100
            recall = tp/(tp+fn)*100
            print('Accuracy: {}%'.format(acc))
            print('Precision: {}%'.format(precision))
            print('Recall: {}%'.format(recall))
            print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

            print(accHistory)

        print(permutationDict)
