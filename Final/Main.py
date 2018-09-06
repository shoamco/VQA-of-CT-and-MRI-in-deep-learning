
"""
This program runs the  whole project:
Extracts from VQA data only the relevant data for us (MRI \ CT),
Then train the model on the training set and get prediction for the test set - which will be our answers data,
 And finally examines the results using metrics : wbss,bleu.
"""


from divide_dic import  Divide_Images
from CnnModels import Get_Predition_of_Train_Model
from Evaluator import EvaluatorAnswers
from formatOfEvaluate import formatToEvaluate
from Writing_Answers import  WritingAnswers

RealAnswers= "VQA Files/VQA_Test.xlsx"# the path to the Real  Answers file of the test-set

# Extracts from VQA data only the relevant data for us (MRI \ CT),and Divide images into folders of train valid test
Divide_Images()#
# rain the model on the training set and get prediction for the test set - which will be our answers data
prediction=Get_Predition_of_Train_Model()
# Write the prediction as the answers in the test file
OurAnswers=WritingAnswers(prediction)
#  converts files of the real Answers and our ansewrs into a format that fit the Evaluate
pathOurAnswers,pathRealAnswers=formatToEvaluate(OurAnswers,RealAnswers)
#  Compare our answers to the real answers in the Evaluator, and print the result of: BLUE WBSS
EvaluatorAnswers(pathOurAnswers,pathRealAnswers)