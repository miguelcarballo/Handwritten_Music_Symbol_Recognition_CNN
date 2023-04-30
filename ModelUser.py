from tqdm import tqdm
import os
import cv2
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from ModelGenerator import *


def create_test_data( dir_test_img, x_size, y_size):
    data = []
    for img in tqdm(os.listdir(dir_test_img)):
        path = os.path.join(dir_test_img, img)
        img_name = img.split('.')[0] 
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        try:
            img_data = cv2.resize(img_data, (x_size,y_size))
        except cv2.error as e:
            continue
        data.append([np.array(img_data), img_name])

    #shuffle(data)
    return data

def use_model(dir_homus_img, dir_test_img, dir_models, name_model, x_size, y_size):  
    #model_trained = train_model(dir_homus_img)
    #name_model_ext = name_model + ".h5"
    net = get_net(x_size, y_size)
    model = tflearn.DNN(net)

    name_model_ext = name_model #+ ".model"
    model.load(os.path.join(dir_models,name_model_ext))

    labels = createLabelAndValArrayFromDir(dir_homus_img)[0]
    
    fig = plt.figure(figsize=(12,12))
    test_data = create_test_data( dir_test_img, x_size, y_size)
    result = []
    for num, data in enumerate(test_data):
        img_data = data[0]
        img_label = data[1]
        y = fig.add_subplot(5,5, num + 1)
        orig = img_data
        data = img_data.reshape(x_size,y_size, 1)
        model_out = model.predict([data])
        prediction =  str(labels[np.argmax(model_out)])
        str_label = img_label + " : " + prediction
        name_and_prediction = [img_label, prediction]
        result.append(name_and_prediction)
        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    print(result)
    plt.show()

