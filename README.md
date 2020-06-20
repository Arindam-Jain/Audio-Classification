# Audio-Classification
# Dataset- 
1045 glass breaking mono audio samples of 6 seconds at 22.5kHz.
1350 baby crying mono audio samples of 6 seconds at 22.5kHz.
1700 noise mono audio samples of 6 seconds at 22.5kHz. 

# Features Extracted- 
64 features using MFCC

# Parameters Used for MFCC- 
n_fft= 2048
hop_length= 512
As n_fft should be greater than sample length and should be a power of 2.

# Creating Dataset
Extracted MFCC features from dataset and created ‘.npy’ files for  training.

# Training Model Used- 
Support Vector Machine (SVM)

# Results:- Features

