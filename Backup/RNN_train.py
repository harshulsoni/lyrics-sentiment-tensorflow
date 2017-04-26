import tensorflow as tf
import numpy as np
import RNN_model
import nltk
import helper_fun
import sys
import random
from gensim.models import word2vec
myRnnModel=RNN_model.build_graph(num_batch=RNN_model.parameters.num_batch, max_sequence_len=RNN_model.parameters.max_sequence_len, hidden_units=RNN_model.parameters.hidden_units, num_classes=RNN_model.parameters.num_classes)


train_x=[]
train_y=[]
seq_len=[]
print ("Beginning Training")

with open('data/pos.txt', "r") as f:
	for line in f:
		cur=helper_fun.paragraph_to_sentencelist(line, remove_stopwords=True)
		train_x+=cur
		train_y.append([0, 1])
		seq_len.append(len(cur[0]))


with open('data/neg.txt', "r") as f:
	for line in f:
		cur=helper_fun.paragraph_to_sentencelist(line, remove_stopwords=True)
		train_x+=cur
		train_y.append([1, 0])
		seq_len.append(len(cur[0]))

sess=tf.Session()
init=tf.initialize_all_variables()
#sess.run(init)
myRnnModel['saver'].restore(sess, "trainedmodels/rnn.model")

for i in range(RNN_model.parameters.num_epoch):
	idx=list(range(len(train_x)))
	random.shuffle(idx)
	train_x=[train_x[i] for i in idx]
	train_y=[train_y[i] for i in idx]
	seq_len=[seq_len[i] for i in idx]
	print ("training epoch", i+1, "of", RNN_model.parameters.num_epoch)
	z=len(train_x)
	j=0
	epoch_loss=0
	while j<z and j+RNN_model.parameters.num_batch<z:
		x_train=train_x[j:j+RNN_model.parameters.num_batch]
		y_train=train_y[j:j+RNN_model.parameters.num_batch]
		seq=seq_len[j:j+RNN_model.parameters.num_batch]
		max_len=max(seq)+1
		j+=RNN_model.parameters.num_batch
		data_x=[]
		for sen in x_train:
			data_x.append(helper_fun.sentence_to_vector(sen, max_len))
		seq=np.array(seq)
		_, loss=sess.run([myRnnModel['minimizer'], myRnnModel['loss']], feed_dict={myRnnModel['x']:data_x, myRnnModel['y']:y_train, myRnnModel['seqlen']:seq})
		epoch_loss+=loss
	print ("Epoch: ", i+1, "completed. Loss: ", epoch_loss)
	if i%10==0:
		myRnnModel['saver'].save(sess, "trainedmodels/rnn.model")
	if epoch_loss<=5.0:
		break
myRnnModel['saver'].save(sess, "trainedmodels/rnn.model")


#best Loss:  37.5073378573