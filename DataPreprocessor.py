import numpy as np
# importing image object from PIL
import math
from PIL import Image, ImageDraw

import os
import cv2
from random import shuffle
from tqdm import tqdm

import shutil


def getDataFromTxt(filename):
    f = open(filename, "r")
    count = 0
    all_lines = []
    label = ""
    for line in f.readlines():
        if(count == 0):
            l = line.strip('\n')
            label = l
        else:
            l = line.strip('\n')
            line_seq = []
            all_coordinates_string = filter(None, l.split(';'))
            #print(all_coordinates_string)
            for coordinate_s in all_coordinates_string:
                coordinate = coordinate_s.split(',')
                coordinate = np.array(coordinate)
                coordinate = coordinate.astype(int)
                line_seq.append(coordinate)
            all_lines.append(line_seq)
        count = count + 1
    f.close()
    data = [label , all_lines]
    return data

def createImgFromData(txt_path, img_path, x_size, y_size):
    data = getDataFromTxt(txt_path)

    #x_size = 150 #pixels
    #y_size = int (1 * x_size)
    center = [int(x_size/2), int(y_size/2)]

    data = getDataCentered(center, data)
    #print(data)
    img = Image.new('RGB', (x_size, y_size))
    all_lines = data[1]; #just data, no label
    for line in all_lines:
        img1 = ImageDraw.Draw(img)
        shape = tuple(map(tuple, line))
        img1.line(shape, (255,255,255), width = 0)
        #img.putpixel((c[0],c[1]),(255,255,255))
    img.save(img_path)

#CENTER THE IMAGE
def getDataCentered(n_center, data): #center is [Cx, Cy]
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for line in data[1]:
        for coordinate in line:
            if(coordinate[0] < min_x):
                min_x = coordinate[0]
            if(coordinate[1] < min_y):
                min_y = coordinate[1]
            if(coordinate[0] > max_x):
                max_x = coordinate[0]
            if(coordinate[1] > max_y):
                max_y = coordinate[1]
    c_center = [min_x +(max_x - min_x)//2, min_y + (max_y - min_y)//2]
    #print("center old image: ", c_center)
    dist = [c_center[0] - n_center[0], c_center[1] - n_center[1]]
    
    for line in data[1]: #move all the coordinates according to the new center
        for coordinate in line:
            coordinate[0] = coordinate[0] - dist[0]
            coordinate[1] = coordinate[1] - dist[1]
    return data

def generateHOMUSimages(dir_homus, dir_homus_img, x_size, y_size):
    #dir_homus = "HOMUS"
    #dir_homus_img = "HOMUS_IMG"

    CHECK_FOLDER_IMG = os.path.isdir(dir_homus_img) 
    if CHECK_FOLDER_IMG: #if there is a folder HOMOS_IMG, delete all the content
        try:
            shutil.rmtree(dir_homus_img)
            print("Deleted directory : ", dir_homus_img)
        except OSError as e:
            print("Error: %s : %s" % (dir_homus_img, e.strerror))
    #if there is not directory for IMG, create it
    os.makedirs(dir_homus_img) #create the directory 
    print("Created directory : ", dir_homus_img)

    for folder_p in tqdm(os.listdir(dir_homus)): #for every folder
        if os.path.isdir(dir_homus + "/" + folder_p): #check if actually is a dir
            for file in os.listdir(dir_homus + "/" + folder_p):  #for 
                if file.endswith('.txt'):
                    path_txt = dir_homus+"/"+folder_p+"/"+file;
                    data = getDataFromTxt(path_txt) #[label, data]
                    #create image, save in folder "label", if does not exist, create.
                    #verify if folder label exists
                    dir_label_img = dir_homus_img+"/"+ data[0] #label = data[0]
                    CHECK_FOLDER = os.path.isdir(dir_label_img) 
                    if not CHECK_FOLDER:
                        os.makedirs(dir_label_img)
                        #print("created folder : ", dir_label)
                    name_file = os.path.splitext(file)[0]+'.png'
                    path_img = dir_label_img + "/" + name_file
                    createImgFromData(path_txt, path_img, x_size, y_size)
    print("All images were successfully generated! Dir: ", dir_homus_img)

