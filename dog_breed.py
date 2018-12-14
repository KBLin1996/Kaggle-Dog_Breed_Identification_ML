import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

INPUT_SIZE = [224, 224, 3]
N_CLASSES = 120
LEARNING_RATE = 2e-5
EPOCHS = 20
BATCH_SIZE = 16
LOAD_PRETRAIN = True

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def train_eval(sess, x_data, y_label, batch_size, train_phase, is_eval, epoch=None):
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    tmp_loss, tmp_acc = 0, 0
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        _, batch_loss, batch_acc = sess.run([train_op, loss, accuracy], feed_dict={x: x_data[start:end], y: y_label[start:end], is_training: train_phase})
        tmp_loss += batch_loss * (end - start)
        tmp_acc += batch_acc * (end - start)
    tmp_loss /= n_sample
    tmp_acc /= n_sample

    if train_phase:
        print('\nepoch: {0}, loss: {1:.4f}, acc: {2:.4f}'.format(epoch+1, tmp_loss, tmp_acc))
    
    return tmp_acc, tmp_loss

def test_eval(sess, x_data, train_phase):
    batch_size = 1
    n_sample = x_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    tmp_pred=[]
    log=[]
    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        tmp_logits = sess.run(logits, feed_dict={x: x_data[start:end], is_training: train_phase})
        tmp=softmax(np.squeeze(tmp_logits))
        tmp_pred.append(tmp)
    tmp_pred = np.array(tmp_pred)

    return tmp_pred


# data preprocess by yourself


if __name__ == '__main__':

    Train_data = pd.read_csv('labels.csv',sep = ',', encoding = 'utf-8')
    Breeds = Train_data['breed']

    sess = tf.Session()
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(Breeds)
    Train_y = label_encoder.transform(Breeds)

    Train_y = tf.one_hot(Train_y, 120, dtype=np.float32)
    Train_y = Train_y.eval(session=sess)

    Source = os.getcwd()
    Train_Path = os.path.join(Source, 'train')
    Test_Path = os.path.join(Source, 'test')

    Train_x = []
    Test_x = []

    x_axis = []
    y_axis = []
    z_axis = []

    for images in os.listdir(Train_Path):
        Image = os.path.join(Train_Path, images)

        Train_Image = cv2.imread(Image)
        Train_Image = cv2.resize(Train_Image, (224, 224), interpolation=cv2.INTER_LANCZOS4)/255
        Train_x.append(Train_Image)

    for images in os.listdir(Test_Path):
        Image = os.path.join(Test_Path, images)
        Test_Image = cv2.imread(Image)
        Test_Image = cv2.resize(Test_Image, (224, 224), interpolation=cv2.INTER_LANCZOS4)/255
        Test_x.append(Test_Image)

    Train_x = np.array(Train_x, dtype = np.float32)
    Test_x = np.array(Test_x, dtype = np.float32)

    x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, N_CLASSES), name='y')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='train_phase')

    logits = model.VGG16(x=x, is_training=is_training, n_classes=N_CLASSES)

    
    with tf.name_scope('LossLayer'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1)), tf.float32))

    restore_variable = [var for var in tf.global_variables() if not var.name.startswith('class/')]
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(restore_variable)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        if LOAD_PRETRAIN:
            sess.run(init)
            saver.restore(sess, 'model/model.ckpt')
        else:
            sess.run(init)

        for i in range(EPOCHS):
            final_acc, final_loss = train_eval(sess=sess, x_data=Train_x, y_label=Train_y, batch_size=BATCH_SIZE, train_phase=True, is_eval=False,epoch=i)
            x_axis.append(i)
            y_axis.append(final_acc)
            z_axis.append(final_loss)

        saver.save(sess, 'model/model.ckpt')
        ans = test_eval(sess=sess, x_data=Test_x, train_phase=False)

        Header = []

        Submit = pd.read_csv('sample_submission.csv',sep = ',', encoding = 'utf-8')
        ID = Submit['id']

        Header = Submit.iloc[0]
        Header = Header.index.values

        Header = Header[1:len(Header)]

        output = pd.DataFrame(ans, index=ID, columns=Header)
        output.to_csv('final.csv')

        plt.plot(x_axis, y_axis, 'g', label='Training Accuracy')
        plt.plot(x_axis, z_axis, 'b', label='Training Loss')

        plt.title('Training Acc/Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Value')

        plt.show()
 
