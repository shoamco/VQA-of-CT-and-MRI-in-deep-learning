
"""
this module Write the prediction as the answers in the test file

"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# The function get prediction and  Write them as the answers in the test file (save in excel file)
# and retern the path to this file
def WritingAnswers(prediction):

    path='VQA Files/OurAnswers.xlsx'# the path to the Our  Answers file of the test-set

    dfANS = pd.read_excel(open('VQA Files/VQA_Test.xlsx', 'rb'), names=['Images', 'Questions', 'Answers'])
    dfANS = dfANS.drop(columns=['Answers'])
    dfANS = pd.merge(dfANS, prediction, on=['Images'])
    writer = ExcelWriter(path)
    dfANS.to_excel(writer, index=False)
    writer.save()
    return path
