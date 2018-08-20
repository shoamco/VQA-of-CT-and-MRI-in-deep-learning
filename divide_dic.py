import pandas as pd
import os
import numpy as np
from os.path import isdir
from os import makedirs
import shutil
from pandas import ExcelWriter
import warnings
warnings.filterwarnings("ignore")


from sklearn.utils import resample

def balance_data(n_samples,imbalance_data):
	df_majority = imbalance_data[imbalance_data.balance == 0]
	df_minority = imbalance_data[imbalance_data.balance == 1]

	# Upsample minority class
	df_minority_upsampled = resample(df_minority,
									 replace=True,  # sample with replacement
									 n_samples=n_samples,  # to match majority class
									 random_state=123)  # reproducible results

	# Combine majority class with upsampled minority class
	df_upsampled = pd.concat([df_majority, df_minority_upsampled])
	# Display new class counts
	# print(df_upsampled.balance.value_counts())
	return df_upsampled

def open_data():
	df = pd.read_csv('InputFiles/dataset.csv',
					 names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns
	VQAM = pd.read_csv('InputFiles/VQAM.csv',
					   names=['Images', 'Questions', 'Answers'])  # open csv file and rename columns

	# dictionary of replaceable words
	replace_dict = {"magnetic resonance imaging": "mri",
					"mri scan": 'mri',
					"MRI": "mri",
					"shows": "show",
					"reveal": "show",
					"demonstrate": "show",
					"CT": "ct",
					"ct scan": "ct",
					"does": "", "do ": "", "the": "",
					# " a ":' ',' is ':' ',
					}
	df.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word
	VQAM.replace(to_replace=replace_dict, inplace=True, regex=True)  # replace word

	# get all the Mri releted rows
	Ct = df[(df['Questions'].str.contains(' ct ') | df['Answers'].str.contains(' ct ')) == True]
	Mri = df[(df['Questions'].str.contains(' mri  ') | df['Answers'].str.contains(' mri ')) == True]

	ImagesMri = ["images\Valid-images\\" + img + ".jpg" for img in Mri['Images']]
	ImagesCt = ["images\Valid-images\\" + img + ".jpg" for img in Ct['Images']]

	sizeMri = len(ImagesMri)
	sizeCt = len(ImagesCt)

	# *************TRAIN - balanced and divide (80%)*******************

	BMri=Mri[ImagesMri[:int(sizeMri * 0.8)]]
	BMri['balance']= np.full(len(BMri), 1) # adding a label for each rows ('1'=mri)
	# save vqa training set in excel

	writer = ExcelWriter('FinelFiles/VQA_TrainingSet.xlsx')
	BMri.to_excel(writer, 'MRI', index=False)
	writer.save()

	BCt = Ct[ImagesCt[:int(sizeCt * 0.8)]]
	BCt['balance'] = np.full(len(BCt), 0)

	df = pd.concat([BMri, BCt])

	df=balance_data(len(BCt),df)####################
	df.set_index('balance')


	# arrays for training after balanced
	ImagesMriTrain = ["images\Train-images\\" + row['Images'] + ".jpg"  for index,row in df.iterrows() if row['balance']== 1 ]
	ImagesCtTrain = ["images\Train-images\\" + row['Images'] + ".jpg"  for index,row in df.iterrows() if row['balance']==0 ]


	TrainData = np.concatenate((ImagesMriTrain, ImagesCtTrain), axis=0, out=None)
	y = np.full((len(ImagesMriTrain)), 0)
	y1 = np.full((len(ImagesCtTrain)), 1)
	TrainLabel = np.concatenate((y, y1), axis=0, out=None)



	# *************VALIDATION - balanced and divide (10%)*******************

	BMri = Mri[ImagesMri[int(sizeMri * 0.8):int(sizeMri * 0.9)]]
	BMri['balance'] = np.full(len(BMri), 1)  # adding a label for each rows ('1'=mri)

	BCt = Ct[ImagesCt[int(sizeCt * 0.8):int(sizeCt * 0.9)]]
	BCt['balance'] = np.full(len(BCt), 0)

	df = pd.concat([BMri, BCt])

	df = balance_data(len(BCt), df)
	df.set_index('balance')

	# arrays for training after balanced
	ImagesMriValid = ["images\Train-images\\" + row['Images'] + ".jpg" for index, row in df.iterrows() if
					  row['balance'] == 1]
	ImagesCtValid = ["images\Train-images\\" + row['Images'] + ".jpg" for index, row in df.iterrows() if
					 row['balance'] == 0]

	validData = np.concatenate((ImagesMriValid, ImagesCtValid), axis=0, out=None)
	y = np.full((len(ImagesMriValid)), 0)
	y1 = np.full((len(ImagesCtValid)), 1)
	validLabel = np.concatenate((y, y1), axis=0, out=None)

	# ****************TEST-   divide (10%)  without balanced*********************************



	testData = np.concatenate((ImagesMri[int(sizeMri * 0.9):], ImagesCt[int(sizeCt * 0.9):]), axis=0, out=None)
	y = np.full((len(ImagesMri[int(sizeMri * 0.9):])), 0)
	y1 = np.full((len(ImagesCt[int(sizeCt * 0.9):])), 1)
	testLabel = np.concatenate((y, y1), axis=0, out=None)

	# *******************


	SizeTrain = len(TrainData)
	SizeValid = len(validData)
	SizeTest = len(testData)

	return (TrainData, TrainLabel, validData, validLabel, SizeTrain, SizeValid, SizeTest, testData, testLabel)


# Divide images into folders of train valid test
# In which each folder has a partition of CT MRI
# (To be able to use the model of keras)

#
def Divide_images_into_folders():
	TrainData, TrainLabel, validData, validLabel, SizeTrain, SizeValid, SizeTest, testData, testLabel = open_data()
	makedirs('data')

	train_folder = 'data/train'
	valid_folder = "data/valid"
	test_folder = 'data/test'





	if isdir(train_folder):  # if directory already exists
		shutil.rmtree(train_folder)
	if isdir(valid_folder):  # if directory already exists
		shutil.rmtree(valid_folder)
	if isdir(test_folder):  # if directory already exists
		shutil.rmtree(test_folder)
	makedirs(train_folder + '/ct/')
	makedirs(train_folder + '/mri/')
	makedirs(valid_folder + '/ct/')
	makedirs(valid_folder + '/mri/')
	makedirs(test_folder + '/ct/')
	makedirs(test_folder + '/mri/')

	for f, i in zip(TrainData, TrainLabel):##train folder
		basename = os.path.basename(f)
		head, tail = os.path.splitext(basename)

		if i == 0:
			dst_dir=train_folder + '/mri/'
			dst_file = os.path.join(dst_dir, basename)
			if not os.path.exists(dst_file):
			    shutil.copy2(f, dst_dir)
			else:
				count = 0
				dst_file1=dst_file
				while os.path.exists(dst_file1):
					count += 1
					dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
				# print 'Renaming %s to %s' % (file, dst_file)
				os.rename(dst_file, dst_file1)
				shutil.copy2(f, dst_dir)

		else:
			dst_dir = train_folder + '/ct/'
			dst_file = os.path.join(dst_dir, basename)
			if not os.path.exists(dst_file):
				shutil.copy2(f, dst_dir)
			else:
				count = 0
				dst_file1 = dst_file
				while os.path.exists(dst_file1):
					count += 1
					dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
				# print 'Renaming %s to %s' % (file, dst_file)
				os.rename(dst_file, dst_file1)
				shutil.copy2(f, dst_dir)

	for f, i in zip(validData, validLabel):##valid folder
		if i == 0:
			dst_dir = valid_folder + '/mri/'
			dst_file = os.path.join(dst_dir, basename)
			if not os.path.exists(dst_file):
				shutil.copy2(f, dst_dir)
			else:
				count = 0
				dst_file1 = dst_file
				while os.path.exists(dst_file1):
					count += 1
					dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
				# print 'Renaming %s to %s' % (file, dst_file)
				os.rename(dst_file, dst_file1)
				shutil.copy2(f, dst_dir)
		else:
			dst_dir = valid_folder + '/ct/'
			dst_file = os.path.join(dst_dir, basename)
			if not os.path.exists(dst_file):
				shutil.copy2(f, dst_dir)
			else:
				count = 0
				dst_file1 = dst_file
				while os.path.exists(dst_file1):
					count += 1
					dst_file1 = os.path.join(dst_dir, '%s-%d%s' % (head, count, tail))
				# print 'Renaming %s to %s' % (file, dst_file)
				os.rename(dst_file, dst_file1)
				shutil.copy2(f, dst_dir)
	for f, i in zip(testData, testLabel):##test folder
		if i == 0:
			shutil.copy2(f, test_folder + '/mri/')
		else:
			shutil.copy2(f, test_folder + '/ct/')


# Data pre-processing and data augmentation
if not os.path.exists('data'):
	Divide_images_into_folders()


