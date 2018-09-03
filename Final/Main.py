from divide_dic import  Divide_Images

from CnnModels import Get_Predition_of_Train_Model
from Evaluator import EvaluatorAnswers
from formatOfEvaluate import formatToEvaluate
from Writing_Answers import  WritingAnswers
RealAnswers= "VQA Files/VQA_Test.xlsx"

Divide_Images()# Divide images into folders of train valid test
prediction=Get_Predition_of_Train_Model()

OurAnswers=WritingAnswers(prediction)
pathOurAnswers,pathRealAnswers=formatToEvaluate(OurAnswers,RealAnswers)
EvaluatorAnswers(pathOurAnswers,pathRealAnswers)