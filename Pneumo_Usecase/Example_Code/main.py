import os
import numpy as np
import glob
from keras.preprocessing.image import load_img, img_to_array

if __name__ == '__main__':
    # Input data files are available in the "./chest_xray" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
    print(os.listdir("./chest_xray"))

    img_size = (224, 224)
    img_array_list = []
    cls_list = []

    train_p_dir = './chest_xray/train/PNEUMONIA'
    img_list = glob.glob(train_p_dir + '/*.jpeg')
    for i in img_list:
        img = load_img(i, color_mode='grayscale', target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_list.append(img_array)
        cls_list.append(1)

    train_n_dir = './chest_xray/train/NORMAL'
    img_list = glob.glob(train_n_dir + '/*.jpeg')
    for i in img_list:
        img = load_img(i, color_mode='grayscale', target_size=(img_size))
        img_array = img_to_array(img) / 255
        img_array_list.append(img_array)
        cls_list.append(0)

    X_train = np.array(img_array_list)
    y_train = np.array(cls_list)
    print(X_train.shape, y_train.shape)