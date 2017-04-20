
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import _pickle as cPickle
import os
import time

img_size_cropped = 32  # v1

batch_size = 500
epoches = 600
beta = 0.27


def unpickle(file):
    with open(file, mode='rb') as file1:
        data = cPickle.load(file1, encoding='bytes')

    return data

def concate_data(data1 ,data2):
    concated = np.concatenate((data1, data2), axis = 0)
    return concated


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



def random_bat(data_x, batch_size):
    num_images = len(data_x)
    idx = np.random.choice(num_images,
                           size= batch_size,
                           replace=False)
    x_batch = data_x[idx, :]
    y_batch = data_label[idx, :]

    return x_batch, y_batch



def random_bat(data_x, data_label):
    num_images = len(data_x)
    idx = np.random.choice(num_images,
                           size= batch_size,
                           replace=False)
    x_batch = data_x[idx, :]
    y_batch = data_label[idx, :]

    return x_batch, y_batch

def L2_regulazier(w_var):

    regulazier = sum([tf.nn.l2_loss(w) for w in w_var])

    return regulazier

def min_random_bat(data, data2):
    num_images = len(data)
    idx = np.random.choice(num_images,
                           size= 100,
                           replace=False)
    x_batch = data[idx, :]
    y_batch = data2[idx, :]

    return x_batch, y_batch


def main():





    # data
    data_batch1 = {}
    data_batch2 = {}
    data_batch3 = {}
    data_batch4 = {}
    data_batch5 = {}


    data_batch1 = unpickle('cifar-10-batches-py/data_batch_1')
    data_batch2 = unpickle('cifar-10-batches-py/data_batch_2')
    data_batch3 = unpickle('cifar-10-batches-py/data_batch_3')
    data_batch4 = unpickle('cifar-10-batches-py/data_batch_4')
    data_batch5 = unpickle('cifar-10-batches-py/data_batch_5')

    train_batch =  {}


    train_data = concate_data(data_batch1[b'data'], data_batch2[b'data'])
    train_data = concate_data(train_data, data_batch3[b'data'])
    train_data = concate_data(train_data, data_batch4[b'data'])
    train_data = concate_data(train_data, data_batch5[b'data'])

    train_data_lables = concate_data(data_batch1[b'labels'], data_batch2[b'labels'])
    train_data_lables = concate_data(train_data_lables, data_batch3[b'labels'])
    train_data_lables = concate_data(train_data_lables, data_batch4[b'labels'])
    train_data_lables = concate_data(train_data_lables, data_batch5[b'labels'])

    train_batch['data'] = train_data
    train_batch['lables'] = train_data_lables


    data_x = train_batch['data']



    data_label = []
    for i in train_batch['lables']:
        blabel = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        blabel[i] = 1.
        data_label.append(blabel)

    data_label = np.array(data_label)

    batch_times = int((len(data_label))//batch_size)

    test_batch = unpickle('cifar-10-batches-py/test_batch')

    test_label = []
    for i in test_batch[b'labels']  :
        blabel = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        blabel[i] = 1.
        test_label.append(blabel)

    test_label = np.array(test_label)


    print('train shape:', data_x.shape)
    print('test shape:' , test_label.shape)
    # data




    x = tf.placeholder(tf.float32, shape=[None, 3072])
    images = tf.reshape(x, [-1,32,32,3])

    #images = pre_process(images=images, training=True)  # crop

    y_ = tf.placeholder(tf.float32, shape=[None, 10])


    W_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])

    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*128])

    W_fc1 = weight_variable([8 * 8 * 128, 3072])
    b_fc1 = bias_variable([3072])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([3072, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    sess = tf.InteractiveSession()

    #                   regulazier              #
    w_list= []
    w_list.append(W_conv1)
    w_list.append(W_conv2)
    w_list.append(W_fc1)
    w_list.append(W_fc2)
    regulazier = L2_regulazier(w_list)
    #                   regulazier              #

    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(0.0001, global_step, 100*epoches, 0.96,staircase= True)
    learning_rate = 0.0001


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) + beta *regulazier)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step = global_step)


    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'cifar10_cnn')

    saver = tf.train.Saver()
    try:
        print("Trying to restore last checkpoint ...")


        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)


        saver.restore(sess, save_path=last_chk_path)


        print("Restored checkpoint from:", last_chk_path)
    except:

        print("Failed to restore checkpoint. Initializing variables instead.")
        sess.run(tf.global_variables_initializer())



    xxxxx,  yyyyy = random_bat(data_x, data_label)
    #print('ran', xxxxx.shape, yyyyy.shape)

    sess.run(tf.global_variables_initializer())
    start_time = time.time()

    train_accuracy = []
    test_accuracy = []
    for j in range(epoches): #iterations

        for i in range(batch_times):
          batch_x , batch_y = random_bat(data_x, data_label)
          train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.7})

        train_acc = accuracy.eval(feed_dict={x: batch_x, y_:  batch_y, keep_prob: 1.0})
        test_acc = accuracy.eval(feed_dict={x: test_batch[b'data'], y_:  test_label, keep_prob: 1.0})

        print('------', j, '----------')
        print('train_acc:', train_acc)
        print('test', test_acc)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    #j = 0
    #best_acc = 0
    #best_j = 0

    '''while(True): #iterations

        for i in range(batch_times):
          batch_x , batch_y = random_bat(data_x, data_label)
          train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.7})

        train_acc = accuracy.eval(feed_dict={x: batch_x, y_:  batch_y, keep_prob: 1.0})
        test_acc = accuracy.eval(feed_dict={x: test_batch[b'data'], y_:  test_label, keep_prob: 1.0})

        print('------', j, '----------')
        print('train_acc:', train_acc)
        print('test', test_acc)

        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)


        if test_acc >= best_acc :
            best_acc = test_acc
            best_j = j

        if test_acc > 0.75 :
            break

        j+=1'''



    saver.save(sess,save_path=save_path)
    end_time = time.time()
    time_dif = end_time - start_time

    f = open('model.txt', 'w')
    f.write('Train Accuray\n')

    for i in train_accuracy:
        f.write(str(i)+'\n')

    f.write('Test Accuray\n')

    for j in test_accuracy:
        f.write(str(j)+'\n')

    f.write('Time:\n')
    f.write(str(time_dif))

    f.close()

    '''#                                mini data                              #
    mini_train = data_x[:9000]
    mini_label = data_label[:9000]
    mini_test = data_x[9000:10000]
    mini_tlabel = data_label[9000:10000]

    sess.run(tf.global_variables_initializer())
    start_time = time.time()

    for j in range(epoches): #iterations

        for i in range(10):
          batch_x , batch_y = min_random_bat(mini_train, mini_label)
          print(global_step.eval())
          if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_:  batch_y , keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("test accuracy %g"%accuracy.eval(feed_dict={x: mini_test , y_: mini_tlabel, keep_prob: 1.0}))
          train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.8})



    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: data_x[9000:], y_: data_label[9000:], keep_prob: 1.0}))
    end_time = time.time()
    time_dif = end_time - start_time
    print('time: --', time_dif)



    #                                mini data                              #'''


if __name__ ==  "__main__":
    main()
