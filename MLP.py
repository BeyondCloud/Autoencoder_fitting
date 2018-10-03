
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random, time
from matplotlib.pyplot import figure

# Config the matplotlib backend as plotting inline in IPython


# In[2]:
def get_gaussian_sum(mu,var,mag):
    sum_all = np.zeros((300,))
    for i in range(5):
        cur = mag[i]*gaussian(np.linspace(1, 300, 300), mu[i],  var[i])
        sum_all+=cur
    return sum_all

class MLP(object):
    def __init__(self,N_in,learning_rate=0.5,n_hidden=[1000,500,250,2],alpha=0.0):
        self.n_features = N_in

        self.weights = None
        self.biases = None

        
        self.graph = tf.Graph() # initialize new grap
        self.build(N_in,learning_rate,n_hidden,alpha) # building graph
        # if sess:
        #     self.sess = sess
        # else:
        self.sess = tf.Session(graph=self.graph) # create session by the graph 
        try:
            self.saver.restore(self.sess, "./model/model.ckpt")
            print('model loaded')
        except:
            self.sess.run(self.init_op)
    def build(self,n_features,learning_rate,n_hidden,alpha):
        with self.graph.as_default():
            ### Input
            self.train_features = tf.placeholder(tf.float32, shape=(None,300))
            self.train_targets  = tf.placeholder(tf.float32, shape=(None,300))

            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss, _ = self.structure(
                                               features=self.train_features,
                                               targets=self.train_targets,
                                               n_hidden=n_hidden)

            # regularization loss
            # weight elimination L2 regularizer
            self.regularizer = tf.reduce_sum([tf.reduce_sum(
                        tf.pow(w,2)/(1+tf.pow(w,2))) for w in self.weights.values()]) \
                    / tf.reduce_sum(
                     [tf.size(w,out_type=tf.float32) for w in self.weights.values()])

            # total loss
            self.loss = self.original_loss + alpha * self.regularizer

            # define training operation
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None,n_features))
            self.new_targets  = tf.placeholder(tf.float32, shape=(None,n_features))
            self.new_y_, self.new_original_loss, self.new_encoder = self.structure(
                                                          features=self.new_features,
                                                          targets=self.new_targets,
                                                          n_hidden=n_hidden)  
            self.new_loss = self.new_original_loss + alpha * self.regularizer

            ### Initialization
            self.init_op = tf.global_variables_initializer()  
            self.saver = tf.train.Saver()
   
    def structure(self,features,targets,n_hidden):
        ### Variable
        if (not self.weights) and (not self.biases):
            self.weights = {}
            self.biases = {}

            n_encoder = [300]+n_hidden
            for i,n in enumerate(n_encoder[:-1]):
                self.weights['encode{}'.format(i+1)] = tf.Variable(tf.truncated_normal(
                        shape=(n,n_encoder[i+1]),stddev=0.1),dtype=tf.float32)
                self.biases['encode{}'.format(i+1)] = tf.Variable(tf.zeros( shape=(n_encoder[i+1]) ),dtype=tf.float32)

            n_decoder = [self.n_features,self.n_features]
            for i,n in enumerate(n_decoder[:-1]):
                self.weights['decode{}'.format(i+1)] = tf.Variable(tf.truncated_normal(
                        shape=(n,n_decoder[i+1]),stddev=0.1),dtype=tf.float32)
                self.biases['decode{}'.format(i+1)] = tf.Variable(tf.zeros( shape=(n_decoder[i+1]) ),dtype=tf.float32)                    

        ### Structure
        activation = tf.nn.relu
        print('feature shape',tf.shape(features))
        encoder = self.getDenseLayer(features,
                                     self.weights['encode1'],
                                     self.biases['encode1'],
                                     activation=activation)

        for i in range(1,len(n_hidden)-1):
            encoder = self.getDenseLayer(encoder,
                        self.weights['encode{}'.format(i+1)],
                        self.biases['encode{}'.format(i+1)],
                        activation=activation)   

        encoder = self.getDenseLayer(encoder,
                        self.weights['encode{}'.format(len(n_hidden))],
                        self.biases['encode{}'.format(len(n_hidden))]) 

        decoder = self.getDenseLayer(encoder,
                                     self.weights['decode1'],
                                     self.biases['decode1'],
                                     activation=activation)

        # for i in range(1,len(n_hidden)-1):
        #     decoder = self.getDenseLayer(decoder,
        #                 self.weights['decode{}'.format(i+1)],
        #                 self.biases['decode{}'.format(i+1)],
        #                 activation=activation) 

        y_ =  self.getDenseLayer(decoder,
                        self.weights['decode1'],
                        self.biases['decode1'],
                        activation=tf.nn.sigmoid)

        N = self.n_features
        print(tf.shape(y_))
        split0, split1, split2 = tf.split(y_, [5, 5, 5], 0)
        print(tf.shape(split0))
        print('---------------------')
        mu = y_[:int(N/3)] * 300 # mean
        var = y_[int(N/3):int(N/3*2)] * 25  #var
        mag = y_[int(N/3*2):] *1 #mag

        final_y = get_gaussian_sum(mu,var,mag)
        loss = tf.reduce_mean(tf.pow(targets - final_y, 2))

        # return (y_,loss,encoder)
        return (final_y,loss,encoder)


    def getDenseLayer(self,input_layer,weight,bias,activation=None):
        x = tf.add(tf.matmul(input_layer,weight),bias)
        if activation:
            x = activation(x)
        return x


    def fit(self,X,Y,epochs=10,validation_data=None,test_data=None,batch_size=None):
        X = self._check_array(X)
        Y = self._check_array(Y)

        N = X.shape[0]
        random.seed(9000)
        if not batch_size: batch_size=N

        
        for epoch in range(epochs):
            print("Epoch %2d/%2d: "%(epoch+1,epochs))
            start_time = time.time()

            # mini-batch gradient descent
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index)>0:
                index_size = len(index)
                batch_index = [index.pop() for _ in range(min(batch_size,index_size))]     

                feed_dict = {self.train_features: X[batch_index,:],
                             self.train_targets: Y[batch_index,:]}
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                print("[%d/%d] loss = %9.4f     " % ( N-len(index), N, loss ), end='\r')


            # evaluate at the end of this epoch
            msg_valid = ""
            if validation_data is not None:
                val_loss = self.evaluate(validation_data[0],validation_data[1])
                msg_valid = ", val_loss = %9.4f" % ( val_loss )

            train_loss = self.evaluate(X,Y)
            print("[%d/%d] %ds loss = %9.4f %s" % ( N, N, time.time()-start_time,
                                                   train_loss, msg_valid ))

        if test_data is not None:
            test_loss = self.evaluate(test_data[0],test_data[1])
            print("test_loss = %9.4f" % (test_loss))
        # self.saver.save(self.sess, "./model/model.ckpt")
        # print("Model saved in path: %s" % "./model/model.ckpt")
    def encode(self,X):
        X = self._check_array(X)
        return self.sess.run(self.new_encoder, feed_dict={self.new_features: X})

    def predict(self,X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})

    def evaluate(self,X,Y):
        X = self._check_array(X)
        return self.sess.run(self.new_loss, feed_dict={self.new_features: X,
                                                       self.new_targets: Y})

    def _check_array(self,ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape)==1: ndarray = np.reshape(ndarray,(1,ndarray.shape[0]))
        return ndarray

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def get_data(n=1):
    sum_all = np.zeros((n,300))
    
    for i in range(n):

        data = np.random.rand(5,3) #mean,var,mag
        data[:,0] *= 200
        data[:,0] +=50
        data[:,1] *= 20
        data[:,1] += 5
    
        for ele in data:
            cur = ele[2]*gaussian(np.linspace(1, 300, 300), ele[0],  ele[1])
            sum_all[i,:]+=cur
        sum_all[i,:] = sum_all[i,:]/max(sum_all[i,:]) 
    return sum_all
# In[3]:


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train_data = mnist.train
# valid_data = mnist.validation
# test_data = mnist.test


# In[4]:

model_1 = MLP( n_features=15,
                     learning_rate= 0.005,
                     n_hidden=[256,64,15],
                     alpha=0.0
                    )
all_train = get_data(1000)
all_valid = get_data(100)
all_test = get_data(100)
model_1.fit(X=all_train,
           Y=all_train,
           epochs=10,
           validation_data=(all_valid,all_valid),
           test_data=(all_test,all_test),
           batch_size = 8,
          )

# save_path = model_1.saver.save(model_1.sess, "./model/model.ckpt")

    
view_sample = 5

f, axarr = plt.subplots(2,view_sample,figsize=(view_sample*3, 2*3),dpi=200)
for i in range(view_sample):
    axarr[0,i].plot(all_test[i])
    axarr[1,i].plot((model_1.predict(all_test[i])).flatten())
plt.tight_layout()
plt.show()

