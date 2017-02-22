import re
import nltk
import parameters
from nltk.corpus import stopwords
from gensim.models import word2vec
w2v=word2vec.Word2Vec.load("trainedmodels/word2vecTrained.mod")

#load punkt tokenizer. punkt = punctuations ('.', ',', '?', ...)	
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')



def sentence_to_wordlist(sentence, remove_stopwords=False):
	
	
	#convert to lower case
	lower_case=sentence.lower()
	
	#split
	words=lower_case.split()

	if remove_stopwords:
		#stopwords
		stops=set(stopwords.words("english"))
		#removing stop words
		words=[w for w in words if not w in stops]

	return words

def paragraph_to_sentencelist(paragraph, remove_stopwords=False ):
	
	#strip whitespace from beginning and end
	paragraph=paragraph.strip()

	#remove non letters
	letters_only=re.sub("[^a-zA-Z]", " ", paragraph)
	letters_only=re.sub(" +", " ", letters_only)

	#use tokenizer to split paragraph
	sentences=tokenizer.tokenize(letters_only)

	sentencelist=[]

	for sentence in sentences:
		if len(sentence)>0:
			sentencelist.append(sentence_to_wordlist(sentence, remove_stopwords))

	return sentencelist

def sentence_to_vector(sen, max_len):
	tmp_x=[]
	for w in sen:
		try:
			tmp_x.append(w2v[w])
		except KeyError:
			tmp_x.append([0]*parameters.feature_len)
	tmp_x+=[[0]*parameters.feature_len]*(max_len-len(tmp_x))
	return tmp_x
