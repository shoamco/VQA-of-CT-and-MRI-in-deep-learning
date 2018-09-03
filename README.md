# VQA of ct and mri in deep-learning

## Visual Question Answering in the Medical Domain - Identifying images of mri and ct using  convolutional neural network 


 

Visual Question Answering is a new and exciting problem that combines natural language processing and computer vision techniques.

Inspired by the recent success of visual question answering in the general domain, we focus on visual question answering in the medical domain.

Given a medical image accompanied with a clinically relevant question, participating systems are tasked with answering the question based on the visual image content.


The general way to solve that kind of problem is by finding identical features between the image that the user inserts and the rest of the images in the training set database, as well as identifying the main words in the question statement and comparing them to other similar questions in the training set database.
In our solution, we worked only on questions whose answers contained MRI or CT words.
We tried to identify the type of image and by doing so to produce an answer (MRI or CT) that would of course be part of the original answer.
This part is a major part of the analysis of the text, because it helps us understand the subject of the answer.
The answer file we provide will be compared with the original answer file (using the code specified above).
 
