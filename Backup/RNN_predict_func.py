import tensorflow as tf
import numpy as np
import RNN_model
import helper_fun
import sys
import os
import random


myRnnModelpredict=RNN_model.build_graph(num_batch=1, max_sequence_len=RNN_model.parameters.max_sequence_len, hidden_units=RNN_model.parameters.hidden_units, num_classes=RNN_model.parameters.num_classes)
sess=tf.Session()
init=tf.initialize_all_variables()
myRnnModelpredict['saver'].restore(sess, "trainedmodels/rnn.model")

def RNN_predict(stringlist):

	data_x=[]
	seq_len=[]

	for line in stringlist:
		cur=helper_fun.paragraph_to_sentencelist(line, remove_stopwords=True)
		data_x+=cur
		seq_len.append(len(cur[0]))

	outputs=[]
	for i in range(len(data_x)):
		seq=seq_len[i:i+1]
		dat=data_x[i:i+1]
		x_data=[]
		for sen in dat:		
			x_data.append(helper_fun.sentence_to_vector(sen, seq[0]+1))
		seq=np.array(seq)
		ans=sess.run(myRnnModelpredict['output'], feed_dict={myRnnModelpredict['x']:x_data, myRnnModelpredict['seqlen']:seq})
		outputs=outputs+list(ans)
	return outputs
	

with open('input/input.txt', "r") as f:
	op=RNN_predict(f)
	print (op)