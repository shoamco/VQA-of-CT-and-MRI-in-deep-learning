"""
Analyze data, filter questions by topic and save them in Excel files
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

df = pd.read_csv('InputFiles/VQAM_Training.csv',names=['Images','Questions','Answers'])#open csv file and rename columns
# df = pd.read_csv('VQAM_Validation.csv',names=['Images', 'Questions','Answers'])

# dictionary of replaceable words
replace_dict = {"magnetic resonance imaging":"mri",
                "mri scan":'mri',
                "MRI":"mri",
                "shows": "show",
                "reveal":"show",
                "demonstrate":"show",
                "CT":"ct",
                "What":"what",
                "ct scan":"ct",
                "does":"","do ":"","the":"",
                    # " a ":' ',' is ':' ',
                }
df.replace(to_replace=replace_dict, inplace=True, regex=True)#replace word
df.replace(to_replace=replace_dict, inplace=True, regex=True)#replace word



#Filter by question words
what=df_ct = df[df['Questions'].str.contains('what')]#
where=df_ct = df[df['Questions'].str.contains('where')]#
who=df_ct = df[df['Questions'].str.contains('who')]#
how=df_ct = df[df['Questions'].str.contains('how')]#
which = df[df['Questions'].str.contains('which')]#
when= df[df['Questions'].str.contains('when')]#
Other=df[~df['Questions'].str.contains('what|where|who|how|which|when')]#

#save in excel file
writer = ExcelWriter('outputFiles/questionWords TrainingSet.xlsx')
# writer = ExcelWriter('questionWords ValidationSet.xlsx')
# writer = ExcelWriter('Analysis2 TrainingSet.xlsx')
# writer = ExcelWriter('Analysis2 ValidationSet.xlsx')
df.to_excel(writer,'all',index=False)
what.to_excel(writer,'what',index=False)
where.to_excel(writer,'where',index=False)
who.to_excel(writer,'who',index=False)
which.to_excel(writer,'which',index=False)
when.to_excel(writer,'when',index=False)
Other.to_excel(writer,'Other',index=False)

writer.save()

#filter
# df_mri = df[df['Questions'].str.contains('mri')]#Only questions about mri
# df_ct = df[df['Questions'].str.contains('ct')]#Only questions about ct
# df_others=df[~df['Questions'].str.contains('mri|ct')]#Questions without mri and ct
#
# df_ct = df[df['Questions'].str.contains('ct')]#Only questions about ct
# df_mriSpine = df[df['Questions'].str.contains('mri')& df['Questions'].str.contains('spine')]#mri spine
#
# df_ctSpine = df[df['Questions'].str.contains('ct')& df['Questions'].str.contains('spine')]#ct spine
# arrow=df[df['Questions'].str.contains(' arrow')]#Only questions about arrow

def exract_data_to_excel():
    df = pd.read_csv('InputFiles/VQAM_Training.csv',
                     names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns
    VQAM = pd.read_csv('InputFiles/VQAM.csv',
                       names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns


    df.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word
    VQAM.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word

    Mri = df[(df['Questions'].str.contains(' mri  ') | df['Answers'].str.contains(' mri ')) == True]
    Ct = df[(df['Questions'].str.contains(' ct ') | df['Answers'].str.contains(' ct ')) == True]
    CtMriVal = VQAM[(~VQAM['Questions'].str.contains(' mri | ct ') & VQAM[
        'Answers'].str.contains('mri| ct')) == True]


    writer = ExcelWriter('outputFiles/MRI CT Answers.xlsx')

    Mri.to_excel(writer, 'ImagesOfMri', index=False)
    Ct.to_excel(writer, 'ImagesOfCt', index=False)

    writer.save()


    writer.save()

    writer = ExcelWriter('outputFiles/MRI CT Answers Validation2.xlsx')
    CtMriVal.to_excel(writer, 'ImagesOfCtMriVal', index=False)
    writer.save()


exract_data_to_excel()

