import os
import requests
from zipfile import ZipFile
import io
import re
import tensorflow as tf
import numpy as np

# code @
# https://github.com/nfmcclure/tensorflow_cookbook/tree/master/10_Taking_TensorFlow_to_Production/05_Production_Example

data_dir = 'temp'
data_file = 'text_data.txt'

storage_dir = 'out'
tensorboard_dir = 'tensorboard'

input_text_node = 'input_text'
keep_prob_node = 'keep_prob'
probability_outputs = 'probability_outputs'

accuracy_ = 'accuracy'
loss_ = 'loss'

MODEL_NAME = 'rnn_text_classifier'

epochs = 100
max_sequence_length = 25
min_word_frequency = 10
embedding_size = 50
rnn_size = 10
batch_size = 250
learning_rate = 0.0005

############################################################
dropout_keep_prob = tf.placeholder(tf.float32, name=keep_prob_node)
############################################################

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('---TTT downloading data from server\npls wait...')
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')

    # Save data to text file
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    print('---TTT loading available data')
    # Open data from text file
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]


# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

############################################################
# Save vocab processor (for loading and future evaluation)
vocab_processor.save(os.path.join(storage_dir, "vocab"))
############################################################

# Shuffle and split data
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)

print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# placeholders
############################################################
x_data = tf.placeholder(tf.int32, [None, max_sequence_length], name=input_text_node)
############################################################
y_output = tf.placeholder(tf.int32, [None])

# embedding
embedded_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1., 1.))
embedding_output = tf.nn.embedding_lookup(embedded_mat, x_data)

# RNN cell
cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.matmul(last, weight) + bias

############################################################
rnn_model_outputs = logits_out

# Prediction
rnn_prediction = tf.nn.softmax(rnn_model_outputs, name=probability_outputs)
############################################################

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out,
                                                        labels=y_output)  # logits=float32, labels=int32
loss = tf.reduce_mean(losses, name=loss_)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32),
                          name=accuracy_)

# Add scalar summaries for Tensorboard
with tf.name_scope('Scalar_Summaries'):
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# Declare summary merging operation
summary_op = tf.summary.merge_all()

# Create a graph/Variable saving/loading operations
saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # Start training
    for epoch in range(epochs):

        # Shuffle training data
        shuffled_ix = np.random.permutation(np.arange(len(x_train)))
        x_train = x_train[shuffled_ix]
        y_train = y_train[shuffled_ix]
        num_batches = int(len(x_train) / batch_size) + 1

        # TO DO CALCULATE GENERATIONS ExACTLY
        for i in range(num_batches):
            # Select train data
            min_ix = i * batch_size
            max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
            x_train_batch = x_train[min_ix:max_ix]
            y_train_batch = y_train[min_ix:max_ix]

            # Run train step
            train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5}
            _, summary = sess.run([train_step, summary_op], feed_dict=train_dict)

            summary_writer = tf.summary.FileWriter(tensorboard_dir)
            summary_writer.add_summary(summary, i)

        # Run loss and accuracy for training
        temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
        train_loss.append(temp_train_loss)
        train_accuracy.append(temp_train_acc)

        # Run Eval Step
        test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob: 1.0}
        temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
        test_loss.append(temp_test_loss)
        test_accuracy.append(temp_test_acc)
        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

    # Save model every epoch
    saver.save(sess, os.path.join(storage_dir, MODEL_NAME + ".ckpt"))

    print('all done! model exported!')
