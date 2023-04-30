import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm
#model training
import warnings
warnings.filterwarnings('ignore')

#import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
  

def createLabelAndValArrayFromDir(dir_homus_img): #returns [["label1", "label2", ], [[1,0,0,..],[0,1,0,...]]]
    all_labels = []
    all_valarrays =[]
    for folder_name in os.listdir(dir_homus_img): #for every folder inside
        if os.path.isdir(dir_homus_img + "/" + folder_name): #check if actually is a dir
            all_labels.append(folder_name) #add to labels
    n_labels = len(all_labels)
    
    for c in range(0, n_labels): #for each label create array value: [1,0,0...]
        valarray = np.zeros(n_labels,dtype=int)
        valarray[c]= 1
        all_valarrays.append(valarray)
    
    labels_and_values = [all_labels, all_valarrays]
    return labels_and_values



def create_data(dir_img, x_size, y_size):
    dir_homus_img = dir_img
    
    labels_and_values = createLabelAndValArrayFromDir(dir_homus_img)
    labels = labels_and_values[0]
    lab_values = labels_and_values[1]
    
    data = []
    for folder in tqdm(os.listdir(dir_homus_img)):
        if os.path.isdir(dir_homus_img + "/" + folder):
            for img in os.listdir(dir_homus_img +"/"+folder):
                path = os.path.join(dir_homus_img,folder, img)
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                try:
                    img_data = cv2.resize(img_data, (x_size,y_size))
                except cv2.error as e:
                    continue
                data.append([np.array(img_data), lab_values[labels.index(folder)]])
    #shuffle(data)
    return data
#print(create_data()[8][0][45]) #see the item 9, the data, row 45

def train_model(dir_homus_img, x_size, y_size):
    warnings.filterwarnings('ignore')
    
    data = create_data(dir_homus_img, x_size, y_size)
    prc_train = 0.8 #80%
    n_train = int(len(data)*prc_train)
    
    train = data[:n_train]
    test = data[n_train:]

    X_train = np.array([i[0] for i in train]).reshape(-1, x_size,y_size, 1)
    y_train = [i[1] for i in train]
    X_test = np.array([i[0] for i in test]).reshape(-1, x_size,y_size, 1)
    y_test = [i[1] for i in test]
    
    #tf.compat.v1.reset_default_graph()
    convnet = get_net(x_size, y_size)
    model = tflearn.DNN(convnet, tensorboard_verbose=1)  
    #model.fit({'input': X_train}, {'targets': y_train}, n_epoch=12, validation_set=({'input': X_test}, {'targets': y_test}),show_metric=True)
    model.fit( X_train, y_train, n_epoch=12, validation_set=(X_test,y_test), snapshot_step=10, show_metric=True, run_id='cnn') 
    #model.save('hope.model') 

    #model.load('hope.model')
    return model

def get_net(x_size, y_size):
    convnet = input_data(shape=[x_size,y_size, 1], name='input') #1 for gray scale img
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 256, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5) #prevent a model from overfitting 
    convnet = fully_connected(convnet, 32, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets') 
    return convnet

def train_and_save_model(dir_homus_img, dir_models, name_model, resized_x, resized_y):
    model_trained = train_model(dir_homus_img, resized_x, resized_y)
    name_model_ext = name_model #+ ".model"
    
    model_trained.save(os.path.join(dir_models,name_model_ext))
    print("Model '" + name_model + "' generated successfully. The data/metadata was saved in " + dir_models + "/")


