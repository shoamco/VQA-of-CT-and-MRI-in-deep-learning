# Final Project's Folder
This folder contains all the final files from our project 



## Explanation of the files:

* The [Main.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/Main.py) -runs the whole project:

  Extracts from VQA data only the relevant data for us (MRI \ CT), Then train the model on the training set and get prediction for the test set - which will be our answers data, And finally examines the results  using metrics : wbss,bleu.

* The [divide_dic.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/divide_dic.py)-Extracts from the original VQA only the data containing MRI/CT.

* The [CnnModels.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/CnnModels.py)-Train  the model (VGG19\VGG16\Xception\InceptionV3\ResNet50) and return the prediction for the test set
  Then divides the images into 3 folders :train,validion,test (Because the generator needs this division) and saved in the folder [data](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/tree/master/Final/data)
  
* The [Evaluator.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/Evaluator.py)-Checking our results in metrics: wbss,bleu.

* The [Writing_Answers.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/Writing_Answers.py)  -Writing the final answers according to the prediction in excel file.
* The [ormatOfEvaluate.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/formatOfEvaluate.py) -  Transfers the files are sent to Evaluator to the appropriate format 
* The [DataAnalysis.py file](https://github.com/shoamco/Visual-Question-Answering-in-the-Medical-Domain/blob/master/Final/DataAnalysis.py) - This file contains a preliminary analysis of the VQA DATA. It is used to understand the subjects of the questions.
    the analysis data  saved in the folder outputFiles.

 
