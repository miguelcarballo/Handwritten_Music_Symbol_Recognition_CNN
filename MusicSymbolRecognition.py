import sys
from DataPreprocessor import *
from ModelGenerator import *
from ModelUser import *
#-------------
dir_homus = "HOMUS"
dir_imgs = "HOMUS_IMG"
dir_models = "MODELS"
dir_test_imgs = "TEST_IMG"
size_x_img = 150
size_y_img = 150

resized_x = 60
resized_y = 60

#-------------
# total arguments
n = len(sys.argv)

if (n <= 1):
    print("Use these options as arguments: ")
    print("     preprocess")
    print("     gen_model 'name of model'")
    print("     use_model 'name of model'")
else:
    option = sys.argv[1]
    if(option == "preprocess"):
        generateHOMUSimages(dir_homus, dir_imgs, size_x_img, size_y_img)
    elif (option == "gen_model"): #just works if the preprocess was made
        name_model = "default_model"
        if(n == 3):
            name_model = sys.argv[2]
        train_and_save_model(dir_imgs, dir_models, name_model, resized_x, resized_y)
    elif (option == "use_model"):
        # use_model(dir_homus_img, dir_test_img, dir_models, name_model, x_size, y_size):  
        if(n == 3):
            name_model = sys.argv[2]
            #try:
            use_model(dir_imgs, dir_test_imgs, dir_models, name_model, resized_x, resized_y)
            #except:
            #    print("error trying to use model: '" + name_model + "'")
        else:
            print("'name of model' is not specified")

    else:
        print("Invalid")


 
