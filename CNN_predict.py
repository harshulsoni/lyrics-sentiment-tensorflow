import tensorflow as tf
import numpy as np
import CNN_model
import helper_fun
import sys
import os
import random


myCnnModelpredict=CNN_model.build_graph(num_batch=1, max_sequence_len=CNN_model.parameters.max_sequence_len, hidden_units=CNN_model.parameters.hidden_units, num_classes=CNN_model.parameters.num_classes)


data_x=[]
seq_len=[]

with open('input/input.txt', "r") as f:
	for line in f:
		cur=helper_fun.paragraph_to_sentencelist(line, remove_stopwords=True)
		data_x+=cur
		seq_len.append(len(cur[0]))

sess=tf.Session()
init=tf.initialize_all_variables()
#sess.run(init)
myCnnModelpredict['saver'].restore(sess, "trainedmodels/cnn.model")

outputs=[]
for i in range(len(data_x)):
	#seq=seq_len[i:i+1]
	dat=data_x[i:i+1]
	x_data=[]
	seq=seq_len[i]
	for sen in dat:		
		x_data.append(helper_fun.sentence_to_vector(sen, seq))
	ans=sess.run(myCnnModelpredict['output'], feed_dict={myCnnModelpredict['x']:x_data, myCnnModelpredict['seqlen']:seq})
	outputs=outputs+list(ans)

#print (outputs)
filename='./output/outputCNN.txt'
dire = os.path.dirname(filename)
if not os.path.exists(dire):
    os.makedirs(dire)
with open(filename, 'w') as f:
	for i in outputs:
		f.write(str(i)+'\n')
