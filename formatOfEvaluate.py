import pandas as pd

def formatToEvaluate(OurAnswers,RealAnswers):


 pathOurAnswers="FinelFiles/Evaluate/OurAnswers.csv"
 pathRealAnswers="FinelFiles/Evaluate/RealAnswers.csv"

 Our = pd.read_excel(open(OurAnswers, 'rb'), names=['Images', 'Questions', 'Answers'])
 Real= pd.read_excel(open(RealAnswers, 'rb'), names=['Images', 'Questions', 'Answers'])

 Our = [str(j + 1) + str(Our['Questions'].iloc[j]) + '\t' + str(Our['Images'].iloc[j]) + '\t' + str(
 Our['Answers'].iloc[j]) for j in range(len(Our['Questions']))]

 Real = [str(j + 1) + str(Real['Questions'].iloc[j]) + '\t' + str(Real['Images'].iloc[j]) + '\t' + str(Real['Answers'].iloc[j]) for j in range(len(Real['Questions']))]

 # "<QA-ID><TAB><Image-ID><TAB><Answer>"
 Our = pd.Series(Our)
 Real = pd.Series(Real)

 Our.replace('"', '')  # replace word
 Real.replace('"', '')  # replace word

 Our.to_csv(pathOurAnswers, index=False)
 Real.to_csv(pathRealAnswers, index=False)
 return (pathOurAnswers,pathRealAnswers)
