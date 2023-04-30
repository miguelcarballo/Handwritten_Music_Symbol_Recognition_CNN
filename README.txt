***********************Handwritten Music Symbol Recognition with CNN***********************
This Python project requires the installation of numpy, cv2, random, tqdm, tflearn and matplotlib to run.

Handwritten Online Musical Symbols (HOMUS) dataset was used to develop this project: https://github.com/apacha/Homus/tree/master/HOMUS
It is necessary to download the folder HOMUS and include it in the folder of the project.

Handwritten_Music_Symbol_Recognition_CNN/
	HOMUS/  --> required folder
	TEST_IMG/  --> images for test
	DataPreprocessor.py
	ModelGenerator.py
	ModelUser.py
	MusicSymbolRecognition.py

In this project, the first step in the handwritten music symbol recognition was addressed: recognize isolated 32 different musical symbols with a good grade of accuracy (more than 90%). In order to do that, it was necessary to get a dataset, preprocess the data, and produce a machine learning model. The dataset used was HOMUS and the method to generate the machine learning model was a Convolutional Neural Network (CNN). 

The CNN model was trained with the 80% of the dataset and evaluated with the 20%. Once the model was produced, it was evaluated with self generated images, giving promising results that might be used in future research.


The following commands should be used:
 
1) python3 MusicSymbolRecognition.py preprocess
2) python3 MusicSymbolRecognition.py gen_model main_model
3) python3 MusicSymbolRecognition.py use_model main_model 

The first command line runs the code to process all the data in the folder HOMUS and will create the HOMUS_IMG folder and all the images generated and grouped by labels.

The second command line runs the code to generate the model named “main_model” and it will save it in the folder MODELS. This might take minutes or even hours depending on the parameters set in ModelGenerator.py. This command is not going to work if there is no data preprocessed (HOMUS_IMG) by the command 1. 

The third command line runs the code to predict the label of all the images (150x150 pixels) included in the folder TEST_IMG using the model named “main_model”. If the model does not exist (in the folder MODELS), it will give an error message. By default it will show a similar plot as Figure 7 and also will show the result as an array in the terminal. This command is not going to work if there is data preprocessed (HOMUS_IMG) by the command 1. 



