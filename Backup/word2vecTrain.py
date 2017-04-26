
import nltk
from nltk.corpus import stopwords
from parameters import *
from gensim.models import word2vec
import helper_fun 
import glob
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


def createmodelandtrain(data, workers=num_workers, size=feature_len, min_count=min_word_count, window=context_len, sample=downsampling):
	#define model and train
	model=word2vec.Word2Vec(data, workers=workers, size=size, min_count=min_count, window=window, sample=sample)

	return model

def finalize_model(model):
	#finalize the model
	#no more training possible after this
	#makes model more memory efficient
	model.init_sims(replace=True)
	return model


print ("Beginning Training")

#load punkt tokenizer. punkt = punctuations ('.', ',', '?', ...)
	
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

data=[]

#files=(glob.glob("./data/*.txt"))
files=[]
for file in files:
	with open(file, "r") as f:
		for line in f:
			data+=helper_fun.paragraph_to_sentencelist(line, tokenizer, remove_stopwords=True)

'''
with open('data/neg.txt', "r") as f:
	for line in f:
		data+=helper_fun.paragraph_to_sentencelist(line, tokenizer, remove_stopwords=True)
'''

#print (len(data))

model=createmodelandtrain(data)
model=finalize_model(model)
model.save("trainedmodels/word2vecTrained.mod")

#print(model.vocab)
#model=word2vec.Word2Vec.load("trainedmodels/word2vecTrained.mod")
#print (model.most_similar("arya"))


