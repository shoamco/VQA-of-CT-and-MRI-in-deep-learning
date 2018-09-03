import pandas as pd





def WritingAnswers(prediction):

    path='VQA Files/OurAnswers.xlsx'

    dfANS = pd.read_excel(open('VQA Files/VQA_Test.xlsx', 'rb'), names=['Images', 'Questions', 'Answers'])
    dfANS = dfANS.drop(columns=['Answers'])
    dfANS = pd.merge(dfANS, prediction, on=['Images'])
    writer = ExcelWriter(path)
    dfANS.to_excel(writer, index=False)
    writer.save()
    return path
