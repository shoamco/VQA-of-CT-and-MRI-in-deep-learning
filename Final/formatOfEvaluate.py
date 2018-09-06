import pandas as pd

"""
The function get: the path to the Real Answers file and the path our answers file 
and converts this file into a format that fit the Evaluate
the format is : <QA-ID><TAB><Image-ID><TAB><Answer>
"""
def formatToEvaluate(OurAnswers,RealAnswers):

 pathOurAnswers="VQA Files/Evaluate/OurAnswers.csv"
 pathRealAnswers="VQA Files/Evaluate/RealAnswers.csv"

 Our = pd.read_excel(open(OurAnswers, 'rb'), names=['Images', 'Questions', 'Answers'])
 Real= pd.read_excel(open(RealAnswers, 'rb'), names=['Images', 'Questions', 'Answers'])

 # converts the files into a format : <QA-ID><TAB><Image-ID><TAB><Answer>

 Our = [str(j + 1) + str(Our['Questions'].iloc[j]) + '\t' + str(Our['Images'].iloc[j]) + '\t' + str(
 Our['Answers'].iloc[j]) for j in range(len(Our['Questions']))]

 Real = [str(j + 1) + str(Real['Questions'].iloc[j]) + '\t' + str(Real['Images'].iloc[j]) + '\t' + str(Real['Answers'].iloc[j]) for j in range(len(Real['Questions']))]


 Our = pd.Series(Our)
 Real = pd.Series(Real)

 Our.replace('"', '')  # replace word
 Real.replace('"', '')  # replace word

 Our.to_csv(pathOurAnswers, index=False)
 Real.to_csv(pathRealAnswers, index=False)
 return (pathOurAnswers,pathRealAnswers)
