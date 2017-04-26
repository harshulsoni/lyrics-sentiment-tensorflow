import multiprocessing
#word2vec parameters

#max number of threads to use
num_workers=multiprocessing.cpu_count()

#vector len
feature_len=200

#num of words of context that the training algo takes into account
context_len=10

#min number of times that a word should appear to be in vocabulary of model
min_word_count=25

#The Google documentation recommends values between .00001 and .001. 
downsampling=1e-3

#number of output classes
num_classes=2

#LSTM Hiddem Units
hidden_units=32

#batch size
num_batch=25

#number of LSTM Layer
num_layers=4

#max sequence length supported by LSTM
max_sequence_len=100

#number of epoch
num_epoch=500
num_epoch_CNN=50