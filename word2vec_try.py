#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os
import zipfile
import collections
import random
import math

import tensorflow as tf
import numpy as np
from six.moves import urllib
from six.moves import xrange

#step1 Download the data
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
	"""Download a file if not present, and make sure it's the right size"""
	if not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print("found and verified", filename)
	else:
		print(statinfo.st_size)
		raise Exception(
			'Failed to verify' + filename + '. Can you get to it with a browser?')
	return filename

filename = maybe_download("text8.zip", 31344016)

#Read the data into a string
def read_data(filename):
	f = zipfile.ZipFile(filename)
	for name in f.namelist():
		return f.read(name).split()
	f.close()

words = read_data(filename)
print("Data size", len(words))

#step2 build the dictionamry and replace rare words with UNK token
vocabulary_size = 50000

def build_dataset(words):
	count = [['UNK, -1']]
	#頻度が上位49999を頻度順に辞書型で並べる
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	count = list(count)
	for word in count:
		dictionary[word[0]] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0 #dictionary['UNK']
			unk_count += 1
		data.append(index) #data = [3,0,3,4,5,6,2,1,6]
	count[0] = unk_count#[['UNK,unk_count']]
	print(count[0])
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words # reduce memory
print("Most common words (+UNK"), count[:5]
print("sample data", data[:10])

data_index = 0

#step3:function to generate a trainig batch for the skip_gram model
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32) #一行
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) #一列
	span = 2 * skip_window + 1# [skip_window target skip_window]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
	print(batch[i], '->', labels[i, 0])
	print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

#step4: build and train a skip_gram model

batch_size = 128
embedding_size = 128 # dimension of the embedding vector
skip_window = 1 # how many words to consider left and right
num_skips = 2 # how many times to reuse an input to generate a label

valid_size = 16 #random set of words to evaluatesimilarity on
valid_window = 100 #only pick dev samples in the head of the distribution
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64#number of negative examples to sample

graph = tf.Graph()

with graph.as_default():

	#input_data
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	with tf.device('/cpu:0'):
		embeddings = tf.Variable(
			tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		#construct the variables for the NCE loss
		nce_weights = tf.Variable(
			tf.truncated_normal([vocabulary_size, embedding_size],
								stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

	loss = tf.reduce_mean(
		tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))

	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(
		normalized_embeddings, valid_dataset)
	similarity = tf.matmul(
		valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")



