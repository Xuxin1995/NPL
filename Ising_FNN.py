"""
@author: Jack Huang
"""

import os,time,math,signal,random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class TransitionNN:
    def __init__(self, train,test, layers):
        self.layers = layers
        self.train = train        
        self.test = test
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session()

        self.keep_prob = tf.placeholder(tf.float32)
        self.input_tf = tf.placeholder(tf.float32, shape=[None, self.layers[0]])
        self.stout_tf =  tf.placeholder(tf.float32, shape=[None, self.layers[-1]])
        self.out_pred = self.neural_net(self.input_tf, self.weights, self.biases) 

        self.test_input_tf = tf.placeholder(tf.float32, shape=[None, self.layers[0]])
        self.test_stout_tf =  tf.placeholder(tf.float32, shape=[None, self.layers[-1]])
        self.test_out_pred = self.neural_net(self.test_input_tf, self.weights, self.biases) 

        # l2_regularizer = tf.contrib.layers.l2_regularizer(0.001)

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out_pred,labels=tf.argmax(self.stout_tf,1)))
        # self.loss = self.cross_entropy + tf.add_n(tf.get_collection('losses'))       
        # tf.add_to_collection('losses',self.loss_mse)
        # self.loss=tf.add_n(tf.get_collection('losses'))

        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.cost)
        self.batch_size = 100000
        # self.n=len(self.input)
        self.saver = tf.train.Saver() # 生成saver

        self.MODEL_SAVE_PATH = "save_model/"
        self.MODEL_NAME = "Ising_Fnn"

        init = tf.global_variables_initializer()
        self.sess.run(init)

        
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for i in range(0,num_layers-1):
            W = tf.Variable(tf.random_normal([layers[i],layers[i+1]]))
            # W = self.get_weight(([layers[i],layers[i+1]]),0.0001)
            b = tf.Variable(tf.zeros([1,layers[i+1]]) + 0.1)
            weights.append(W)
            biases.append(b)
        return weights,biases

    # def get_weight(self,shape,lambda1):
    #     W = tf.Variable(tf.random_normal(shape),tf.float32)
    #     tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambda1)(W))
    #     return W

    def neural_net(self, inputs, weights, biases):
        num_layers = len(weights) + 1 
        h = inputs
        for l in range(num_layers - 2):
            W = weights[l]
            b = biases[l]
            h = tf.nn.sigmoid(tf.add(tf.matmul(h,W),b))
            h=tf.nn.dropout(h,self.keep_prob)
        W = weights[-1]
        b = biases[-1]
        h = tf.add(tf.matmul(h,W),b)
        return h 

    def callback(self, loss):
        print('Loss: %e' %(loss))

    def compute_accuracy(self, test_inputs, test_stouts):
        correct_prediction = tf.equal(tf.argmax(self.test_out_pred,1),tf.argmax(self.test_stout_tf,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.test_input_tf: test_inputs, self.test_stout_tf: test_stouts,self.keep_prob:1} )
        return result 

    def training(self, nIter):
        labelBits = 2
        labelBFlag = -1*labelBits
        # train_i = self.train[:, :labelBFlag]
        # train_l = self.train[:, labelBFlag:]
        test_i = self.test[:, :labelBFlag]
        test_l = self.test[:, labelBFlag:]
        
        ckpt = tf.train.get_checkpoint_state(self.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            print("Model Restored")
    
        start_time = time.time()
        it=1
        while it < nIter:
            train_loss = 0
            np.random.shuffle(self.train)

            train_i = self.train[:, :labelBFlag]
            train_l = self.train[:, labelBFlag:]
            # tf_train_dict = {self.input_tf: train_i, self.stout_tf: train_l,self.keep_prob:0.5}
            mini_batches_i = [train_i[k:k+self.batch_size] for k in range(0,len(train_i), self.batch_size)]
            mini_batches_sto = [train_l[k:k+self.batch_size] for k in range(0,len(train_l), self.batch_size)]

            # Use every batch to update weights and biases
            for i in range(len(mini_batches_i)):    
                batch_dict = {self.input_tf: mini_batches_i[i], self.stout_tf: mini_batches_sto[i],self.keep_prob:0.5}
                _,l = self.sess.run([self.optimizer, self.cost], batch_dict)
                train_loss += l / len(mini_batches_i)

            if it % 2000 == 0:
                self.saver.save(self.sess, os.path.join(self.MODEL_SAVE_PATH, self.MODEL_NAME),global_step=it)

            if it % 200 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, train_loss, elapsed))
                start_time = time.time()
            it=it+1

        test_acc = []
        test_acc2 = []
        N_test = 1000
        for i in range(int(len(test_i)/N_test)):
            temp = self.compute_accuracy(test_i[i*N_test:(i+1)*N_test], test_l[i*N_test:(i+1)*N_test])
            test_acc.append(temp)
            test_acc2.append(1-temp)
            np.savetxt('outputs.txt', test_acc) 
            np.savetxt('outputs2.txt', test_acc2) 


#############################
if __name__ == "__main__":

    layers = [400,200,2]
    Data_all = np.load('./L20_40000/all_images.npy')
    print(np.array(Data_all).shape)

    n = len(Data_all)
    np.random.shuffle(Data_all[0])
    Data_train = Data_all[0][0:39000]  ####train_data
    Data_test = Data_all[0][39000:40000]  ######test_data

    for i in range(1,n):
        np.random.shuffle(Data_all[i])
        if i < 19 or i > 28:
            # temp_1 = Data_all[i*1100:i*1100+1000]
            temp_1 = Data_all[i][0:39000]
            temp_2 = Data_all[i][39000:40000] 
            # print(len(temp_1))
            # Data_train.append(temp_1)
            Data_train = np.concatenate((Data_train,temp_1),axis=0)
            Data_test = np.concatenate((Data_test,temp_2),axis=0)
            # temp_2 = Data_all[i*1100+1000:(i+1)*1100]
            # temp_2 = Data_all[i][1000:1100]
            # # Data_test.append(temp_2)
            # Data_test = np.concatenate((Data_test,temp_2),axis=0)
        else:
            temp_2 = Data_all[i][0:1000] 
            Data_test = np.concatenate((Data_test,temp_2),axis=0)

    print(Data_train.shape)
    print(Data_test.shape)

    #### make labels
    gap_num = 19

    A_label = np.array([[1.,0.]])
    B_label = np.array([[0.,1.]])
    # # for i in range(13):
    N_A = 39000*gap_num
    N_B = len(Data_train) - N_A
    A_labels = np.repeat(A_label, N_A, axis=0)
    B_labels = np.repeat(B_label, N_B, axis=0)
    train_labels = np.concatenate((A_labels,B_labels),axis=0)
    print(len(A_labels),len(B_labels))

    N_A =  len(Data_test)
    N_B = 0
    A_labels = np.repeat(A_label, N_A, axis=0)
    B_labels = np.repeat(B_label, N_B, axis=0)
    test_labels = np.concatenate((A_labels,B_labels),axis=0)
    print(len(A_labels),len(B_labels))

    train_data = np.concatenate((Data_train,train_labels),axis=1)
    test_data = np.concatenate((Data_test,test_labels),axis=1)
    model = TransitionNN(train_data,test_data, layers)
    model.training(10001)

