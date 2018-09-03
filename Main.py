from divide_dic import  Divide_Images

from TransferLearningVGG19 import Writing_Answers_according_the_predictions_of_trained_Model
from Evaluator import EvaluatorAnswers
from formatOfEvaluate import formatToEvaluate

RealAnswers= "FinelFiles/VQA_Test.xlsx"

Divide_Images()# Divide images into folders of train valid test
OurAnswers=Writing_Answers_according_the_predictions_of_trained_Model()
pathOurAnswers,pathRealAnswers=formatToEvaluate(OurAnswers,RealAnswers)
EvaluatorAnswers(pathOurAnswers,pathRealAnswers)