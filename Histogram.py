"""
Initial analysis of MRI / CT image type
Using an average histogram
(Only for questions that do not include MRI / CT)
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

what=df_ct = df[df['Questions'].str.contains('what')]#
data=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what'))==True ]#
data2=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') &df['Answers'].str.contains('mri|ct') )==True ]#
data3=df[(~df['Questions'].str.contains('mri|ct') & df['Questions'].str.contains('what') & (df['Answers'].str.contains('mri|ct')==False) )==True ]#
writer = ExcelWriter('outputFiles/questions without CT MRI.xlsx')
data.to_excel(writer,'All Answers',index=False)
data2.to_excel(writer,'only MRI CT in Answers',index=False)
data3.to_excel(writer,'not MRI CT  in Answers',index=False)

writer.save()



