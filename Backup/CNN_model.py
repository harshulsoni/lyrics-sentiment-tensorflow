import tensorflow as tf
import numpy as np
import parameters 


def build_graph(num_batch,max_sequence_len, hidden_units, num_classes):

	# [batch_size, sequence_size, feature_len ]
	x=tf.placeholder(tf.float32, [num_batch, None, parameters.feature_len])
	y=tf.placeholder(tf.float32, [num_batch, num_classes])
	seqlen = tf.placeholder(tf.int32)
	keep_prob = tf.constant(0.8)
	x_image=tf.reshape(x, [num_batch, seqlen, parameters.feature_len, 1])


	W_filter1 = tf.Variable(tf.truncated_normal([5, parameters.feature_len, 1, 40], stddev=0.01))
	b_filter1 = tf.Variable(tf.truncated_normal([40], stddev=0.01))
	W_filter2 = tf.Variable(tf.truncated_normal([3, parameters.feature_len, 1, 40], stddev=0.01))
	b_filter2 = tf.Variable(tf.truncated_normal([40], stddev=0.01))
	W_filter3 = tf.Variable(tf.truncated_normal([2, parameters.feature_len, 1, 20], stddev=0.01))
	b_filter3 = tf.Variable(tf.truncated_normal([20], stddev=0.01))

	h_conv1=tf.nn.relu(tf.nn.conv2d(x_image, W_filter1, strides=[1, 1, 1, 1], padding='VALID')+b_filter1)
	h_conv1_flat=tf.reshape(h_conv1, [-1, 40,seqlen-5+1])
	h_pool1=tf.reduce_max(h_conv1_flat, 2)
	#h_pool1=tf.nn.max_pool(h_conv1, ksize=[1,seqlen-5+1,1, 1], strides=[1, 1, 1, 1], padding='VALID')
	#h_pool1_flat=tf.reshape(h_pool1, [-1, 40])
	h_conv2=tf.nn.relu(tf.nn.conv2d(x_image, W_filter2, strides=[1, 1, 1, 1], padding='VALID')+b_filter2)
	h_conv2_flat=tf.reshape(h_conv2, [-1, 40,seqlen-3+1])
	h_pool2=tf.reduce_max(h_conv2_flat, 2)
	#h_pool2=tf.nn.max_pool(h_conv2, ksize=[1, seqlen-3+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
	#h_pool2_flat=tf.reshape(h_pool2, [-1, 40])
	h_conv3=tf.nn.relu(tf.nn.conv2d(x_image, W_filter3, strides=[1, 1, 1, 1], padding='VALID')+b_filter3)
	h_conv3_flat=tf.reshape(h_conv3, [-1, 20,seqlen-2+1])
	h_pool3=tf.reduce_max(h_conv3_flat, 2)
	#h_pool3=tf.nn.max_pool(h_conv3, ksize=[1, seqlen-2+1,1, 1], strides=[1, 1, 1, 1], padding='VALID')
	#h_pool3_flat=tf.reshape(h_pool3, [-1, 20])
	intermediate_rep=tf.concat(1,[h_pool1, h_pool2, h_pool3])
	#print (int(intermediate_rep.get_shape()[1]))
	#print ([int(x) for x in intermediate_rep.get_shape()])

	#WL1=tf.Variable(tf.truncated_normal([4*4*64, 1024], stddev=0.01))
	WL1=tf.Variable(tf.truncated_normal([100, 1024], stddev=0.01))
	BL1=tf.Variable(tf.truncated_normal([1024], stddev=0.01))
	#WL2=tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.01))
	#BL2=tf.Variable(tf.truncated_normal([1024], stddev=0.01))
	WL3=tf.Variable(tf.truncated_normal([1024, num_classes], stddev=0.01))
	BL3=tf.Variable(tf.truncated_normal([num_classes], stddev=0.01))

	#Model
	outputL1=tf.add(tf.matmul(intermediate_rep, WL1),BL1)
	outputL1=tf.nn.relu(outputL1)
	outputL1_drop = tf.nn.dropout(outputL1, keep_prob)
	#outputL2=tf.add(tf.matmul(outputL1_drop, WL2),BL2)
	#outputL2=tf.nn.relu(outputL2)
	#outputL2_drop = tf.nn.dropout(outputL2, keep_prob)
	outputL3=tf.add(tf.matmul(outputL1_drop, WL3), BL3)
	
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputL3, y))
	predictions=tf.nn.softmax(outputL3)
	output=tf.argmax(predictions, 1)
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

#myRnnModel=build_graph(num_batch=parameters.num_batch, max_sequence_len=parameters.max_sequence_len, hidden_units=parameters.hidden_units, num_classes=parameters.num_classes)