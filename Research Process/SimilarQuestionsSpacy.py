"""
The program checks for each question in the test file which question is most similar in the training file- using SPACY
And returns for it the first answer of the most similar question
Finally,  create a CSV file in the format that corresponds to the evaluation code
"""
import csv
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import spacy
import nltk
# nltk.download('stopwords')
import time




df = pd.read_csv('InputFiles/dataset.csv',names=['Images', 'Questions', 'Answers'])#open csv file and rename columns
predictions = pd.read_csv('InputFiles/VQAM.csv',names=['Images', 'Questions','Answers'])
# dictionary of replaceable words
replace_dict = {"magnetic resonance imaging":"mri",
                "mri scan":'mri',
                "MRI":"mri",
                "shows": "show",
                "reveal":"show",
                "demonstrate":"show",
                "CT":"ct",
                "ct scan":"ct",
                "does":"","do ":"","the":"",
                    # " a ":' ',' is ':' ',
                }
df.replace(to_replace=replace_dict, inplace=True, regex=True)#replace word
predictions.replace(to_replace=replace_dict, inplace=True, regex=True)#replace word


nlp = spacy.load('en_core_web_sm')

#
# start_time = time.time()
# df['nlpQuestions']=[nlp(q) for q in df['Questions']]
# predictions['nlpQuestions']=[nlp(q) for q in predictions['Questions']]
# print(time.time() - start_time)
#
#
# #calculation average Score
# sumScore=0;
# Answers2 = len(predictions['Questions'])
# # for q1 in predictions['nlpQuestions']:
# for i in range(len(predictions['nlpQuestions'])):
#     q1=predictions['nlpQuestions'].iloc[i]
#     maxSimilarity = max([q1.similarity(q) for q in df['nlpQuestions']])
#     ans=[df['Answers'].iloc[j] for j in range(len(df['nlpQuestions'])) if q1.similarity(df['nlpQuestions'].iloc[j])==maxSimilarity]
#     predictions.loc[i,'Answers2'] = ans[0]
#
#
#
#
# res=[str(j+1)+str(predictions['Questions'].iloc[j])+'\t'+str(predictions['Images'].iloc[j])+'\t'+str(predictions['Answers2'].iloc[j] )for j in range(len(predictions['Questions']))]
#
# res2=[str(j+1)+str(predictions['Questions'].iloc[j])+'\t'+str(predictions['Images'].iloc[j])+'\t'+str(predictions['Answers'].iloc[j] )for j in range(len(predictions['Questions']))]
# #"<QA-ID><TAB><Image-ID><TAB><Answer>"
# myRes= pd.Series(res)
# myRes2= pd.Series(res2)
#
# myRes.replace('"', '')#replace word
# myRes2.replace('"', '')#replace word
#
# myRes.to_csv('outputFiles/res.csv',index=False)
# myRes2.to_csv('outputFiles/gt.csv',index=False)
#

q1=nlp('mri')
q=nlp('brain mri')
print(q1.similarity(q))