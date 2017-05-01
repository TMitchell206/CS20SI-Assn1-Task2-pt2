import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from PIL import Image, ImageFilter
from scipy import misc, ndimage
import numpy as np
import pandas as pd

#File location parameters:
dataset_path = '../notMNIST/Lists/'
test_file = 'test-data.csv'
train_file = 'train-data.csv'

#Image parameters:
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1
NUM_EPOCHES = 2

#MODEL PARAMETERS:
BATCH_SIZE = 3
HIDDEN_UNITS = 15
NUM_CLASSES = 10

def encode_label(label):
    if   'A' in label: return [1,0,0,0,0,0,0,0,0,0]
    elif 'B' in label: return [0,1,0,0,0,0,0,0,0,0]
    elif 'C' in label: return [0,0,1,0,0,0,0,0,0,0]
    elif 'D' in label: return [0,0,0,1,0,0,0,0,0,0]
    elif 'E' in label: return [0,0,0,0,1,0,0,0,0,0]
    elif 'F' in label: return [0,0,0,0,0,1,0,0,0,0]
    elif 'G' in label: return [0,0,0,0,0,0,1,0,0,0]
    elif 'H' in label: return [0,0,0,0,0,0,0,1,0,0]
    elif 'I' in label: return [0,0,0,0,0,0,0,0,1,0]
    else:              return [0,0,0,0,0,0,0,0,0,1]

def decode_to_letter(vector):
    argmax = np.argmax(vector)
    if   argmax == 0: return 'A'
    elif argmax == 1: return 'B'
    elif argmax == 2: return 'C'
    elif argmax == 3: return 'D'
    elif argmax == 4: return 'E'
    elif argmax == 5: return 'F'
    elif argmax == 6: return 'G'
    elif argmax == 7: return 'H'
    elif argmax == 8: return 'I'
    else:             return 'J'

def decode_to_letter_batch(batch):
    letters = [decode_to_letter(b) for b in batch]
    return letters

def read_data_file(file):
    files = []
    labels = []
    f = open(file, 'r')
    next(f) #skip header
    for line in f:
        filename, label = line.split(',')
        files.append(filename)
        labels.append(encode_label(label))
    return files, labels

def gen_tensor(data, data_type):
    return ops.convert_to_tensor(data, dtype=data_type)

#Read in CSVs:
img_train, lbl_train = read_data_file(dataset_path+train_file)
img_test, lbl_test = read_data_file(dataset_path+test_file)
train_egs = len(img_train)
test_egs = len(img_test)

#train_batch_cap = int(len(img_train)/BATCH_SIZE)
#test_batch_cap = int(len(img_test)/BATCH_SIZE)

#Convert inputs to tensors
img_train = gen_tensor(img_train, dtypes.string) #images are filepaths
lbl_train = gen_tensor(lbl_train, dtypes.float32)
img_test = gen_tensor(img_test, dtypes.string) #images are filepaths
lbl_test = gen_tensor(lbl_test, dtypes.float32)

#Init queues
#queue_train = tf.train.slice_input_producer([img_train, lbl_train], shuffle=False, capacity=train_batch_cap)
#queue_test = tf.train.slice_input_producer([img_test, lbl_test], shuffle=False, capacity=test_batch_cap)
queue_train = tf.train.slice_input_producer([img_train, lbl_train], shuffle=False)
queue_test = tf.train.slice_input_producer([img_test, lbl_test], shuffle=False)

file_content = tf.read_file(queue_train[0])
img_train = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
lbl_train = queue_train[1]

file_content = tf.read_file(queue_test[0])
img_test = tf.image.decode_png(file_content, channels=NUM_CHANNELS)
lbl_test = queue_test[1]

img_train.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
lbl_train.set_shape([10])
img_test.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
lbl_test.set_shape([10])

#img_train_batch, lbl_train_batch = tf.train.batch([img_train, lbl_train], batch_size=BATCH_SIZE, capacity=train_batch_cap, allow_smaller_final_batch=True)
#img_test_batch, lbl_test_batch = tf.train.batch([img_test, lbl_test], batch_size=BATCH_SIZE, capacity=test_batch_cap, allow_smaller_final_batch=True)
img_train_batch, lbl_train_batch = tf.train.batch([img_train, lbl_train], batch_size=BATCH_SIZE, allow_smaller_final_batch=True)
img_test_batch, lbl_test_batch = tf.train.batch([img_test, lbl_test], batch_size=BATCH_SIZE, allow_smaller_final_batch=True)

weights = {
    'w1': tf.Variable(tf.random_normal([IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS, HIDDEN_UNITS], stddev=0.1)),
    'w2': tf.Variable(tf.random_normal([HIDDEN_UNITS, NUM_CLASSES], stddev=0.1))
    }

biases = {
    'b1': tf.Variable(tf.random_normal([HIDDEN_UNITS], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([NUM_CLASSES], stddev=0.1))
}

def layer(x, W, b):
    x = tf.matmul(x, W)
    return tf.nn.bias_add(x, b)

def sigmoid_layer(x, W, b):
    y = layer(x, W, b)
    return tf.nn.sigmoid(y)

def nn_model(x_input, weights, biases):

    x_flat = tf.reshape(x_input, shape=[-1, IMAGE_HEIGHT*IMAGE_WIDTH*NUM_CHANNELS])
    hidden = layer(x_flat, weights['w1'], biases['b1'])
    out = tf.nn.softmax(sigmoid_layer(hidden, weights['w2'], biases['b2']))
    return out

x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
pred = nn_model(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.pow(y - pred,2))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Evaluate model
cost = tf.reduce_mean(tf.pow(y - pred,2))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:

    #init variables:
    sess.run(tf.global_variables_initializer())

    #init queue threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #print meta-data
    print 'Number of Training examples:', train_egs
    print 'Number of Training batchs:', int(train_egs/BATCH_SIZE)
    print 'Number of Testing examples:', test_egs
    print 'Number of Testing batchs:', int(test_egs/BATCH_SIZE)
    print "Training started..."

    for k in range(NUM_EPOCHES):
        t_cost = 0.
        t_acc = 0.
        #test_acc = 0.

        #print '\t Epoch:', k+1

        num_batches = int(train_egs/BATCH_SIZE)
        for i in range(num_batches):

            avg_cost = 0.
            x_batch, y_batch = sess.run([img_train_batch, lbl_train_batch])
            x_batch = x_batch/255.
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
            t_cost += sess.run(cost, feed_dict={x: x_batch, y: y_batch})/BATCH_SIZE
            t_acc += sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})/BATCH_SIZE
            predicts = sess.run(pred, feed_dict={x: x_batch})

        print 'EPOCH:', k+1
        print '\t Cost: \t', np.round(t_cost/num_batches,5)
        print '\t Acc: \t', np.round(t_acc/num_batches,5)

    print "Training Completed!"
    print "Running Test Cases..."

    test_acc = 0.
    num_batches = int(test_egs/BATCH_SIZE)
    print 'Testing results: '
    for i in range(num_batches):
        x_batch, y_batch = sess.run([img_test_batch, lbl_test_batch])
        test_acc += sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})/(BATCH_SIZE)
    print '\t Acc:', np.round(test_acc/num_batches, 5)

    'Starting dynamic testing: '
    while True:
        print "Enter absolute image filepath to test network:"
        print "Type 'end' to exit testing."
        var = raw_input('>')
        if var == 'end':
            break
        else:
            var = str(var).strip()
            temp_img = Image.open(var)
            temp_img.show()

            temp_img = misc.imread(var)
            new_img = []
            for row in temp_img:
                new_row = [ [pixel] for pixel in row ]
                new_img.append(new_row)

            img = np.array([new_img])/255.

            sm_vector = sess.run(pred, feed_dict={x: img})
            print "Network's best estimate is:", decode_to_letter(sm_vector)

	# stop queue threads and properly close the Session
    coord.request_stop()
    coord.join(threads)
    sess.close()
