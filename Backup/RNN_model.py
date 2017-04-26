import tensorflow as tf
import numpy as np
import parameters 


def build_graph(num_batch,max_sequence_len, hidden_units, num_classes):

	# [batch_size, sequence_size, feature_len ]
	x=tf.placeholder(tf.float32, [num_batch, None, parameters.feature_len])
	y=tf.placeholder(tf.float32, [num_batch, num_classes])
	seqlen = tf.placeholder(tf.int32, [num_batch])
	cell=tf.nn.rnn_cell.LSTMCell(hidden_units, state_is_tuple=True)
	cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.9, output_keep_prob=0.9)
	cell=tf.nn.rnn_cell.MultiRNNCell([cell]*parameters.num_layers, state_is_tuple=True)
	init_state=cell.zero_state(num_batch, tf.float32)
	#calculate LSTM output
	val, final_state =tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=seqlen, initial_state=init_state)
	

	#transpose to switch batch_size with sequence_size 
	#val= tf.transpose(val, [1, 0, 2])

	#since only last output of every batch is required
	#last = tf.gather(val, int(val.get_shape()[0]) - 1)
	idx = tf.range(num_batch)*tf.shape(val)[1] + (seqlen - 1)
	last = tf.gather(tf.reshape(val, [-1, hidden_units]), idx)

	weight = tf.Variable(tf.truncated_normal([hidden_units, num_classes], stddev=0.01))
	bias=tf.Variable(tf.truncated_normal([num_classes], stddev=0.01))
	logits=tf.matmul(last, weight)+bias
	predictions=tf.nn.softmax(logits)
	output=tf.argmax(predictions, 1)
	#y_reshaped=tf.reshape(y, [-1])
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
	optimizer=tf.train.AdamOptimizer()
	minimizer=optimizer.minimize(loss)
	saver=tf.train.Saver()
	return dict(
        x = x,
        y = y,
        seqlen = seqlen,
        loss = loss,
        output=output,
        minimizer = minimizer,
        predictions = predictions,
        saver=saver
    )

