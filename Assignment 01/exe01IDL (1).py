#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
tf.__version__


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from datasets import MNISTDataset


# In[3]:


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(train_images[15], cmap='Greys_r')
data = MNISTDataset(train_images.reshape([-1, 784]), train_labels,
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size = 128)


# # Setting up parameters

# In[ ]:


train_steps = 1000
learning_rate = 1e-4
n_input = 28*28  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)


# In[ ]:


weights = {
    'w1': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_input, n_hidden1]).astype(np.float32)),
    'w2': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden1, n_hidden2]).astype(np.float32)),
    'w3': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden2, n_hidden3]).astype(np.float32)),
    'out':tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden3, n_output]).astype(np.float32)),
}


# In[ ]:


biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}


# In[7]:


get_ipython().run_line_magic('time', '')
losses = []
accs = []
for step in range(train_steps):
  img_batch, lbl_batch = data.next_batch()
  with tf.GradientTape() as tape:
    layer_1 = tf.add(tf.matmul(img_batch, weights['w1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    logits = tf.matmul(layer_3, weights['out']) + biases['out']
    xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = lbl_batch))
  var_list = [weights['w1'],weights['w2'], weights['w3'],
              weights['out'],biases['b1'],biases['b2'],
              biases['b3'],biases['out']]
  grads = tape.gradient(xent, var_list)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  optimizer.apply_gradients(zip(grads,var_list))

  
  if not step % 50:
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                                  tf.float32))
    losses.append(xent)
    accs.append(acc)
    print('Loss: {} Accuracy: {}'.format(xent, acc))


# In[8]:


test_layer_1 = tf.add(tf.matmul(data.test_data, weights['w1']), biases['b1'])
test_layer_2 = tf.add(tf.matmul(test_layer_1, weights['w2']), biases['b2'])
test_layer_3 = tf.add(tf.matmul(test_layer_2, weights['w3']), biases['b3'])
test_out = tf.matmul(test_layer_3, weights['out']) + biases['out']
test_preds = tf.argmax(test_out, axis=1, output_type=tf.int32)

test_acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),
                             tf.float32))
print('Accuracy: ')
print(int(test_acc.numpy()*100))
print("%")


# # Visulaizing loss and accuracy

# In[9]:


plt.plot(losses, 'g', label='loss')
plt.plot(accs , 'r',label='accuracy')
plt.plot(test_acc, '+' ,label = 'test_accuracy')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





# # Fashion Mnist

# In[10]:


get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
tf.__version__


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from datasets import MNISTDataset


# In[12]:


mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(train_images[15], cmap='Greys_r')
data = MNISTDataset(train_images.reshape([-1, 784]), train_labels,
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size = 128)


# # Setting up parameters

# In[ ]:


train_steps = 1000
learning_rate = 1e-4
n_input = 28*28  # input layer (28x28 pixels)
n_hidden1 = 1024  # 1st hidden layer
n_hidden2 = 512  # 2nd hidden layer
n_hidden3 = 256  # 3rd hidden layer
n_hidden4 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)


# In[ ]:


weights = {
    'w1': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_input, n_hidden1]).astype(np.float32)),
    'w2': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden1, n_hidden2]).astype(np.float32)),
    'w3': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden2, n_hidden3]).astype(np.float32)),
    'w4': tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden3, n_hidden4]).astype(np.float32)),
    'out':tf.Variable(np.random.uniform (low=-0.1, high=0.1, 
                                         size = [n_hidden4, n_output]).astype(np.float32)),
}


# In[ ]:


biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'b4': tf.Variable(tf.constant(0.1, shape=[n_hidden4])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}


# In[16]:


get_ipython().run_line_magic('time', '')
losses = []
accs = []
for step in range(train_steps):
  img_batch, lbl_batch = data.next_batch()
  with tf.GradientTape() as tape:
    layer_1 = tf.add(tf.matmul(img_batch, weights['w1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    layer_4 = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
    logits = tf.matmul(layer_4, weights['out']) + biases['out']
    xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = lbl_batch))
  var_list = [weights['w1'],weights['w2'], weights['w3'],weights['w4'],
              weights['out'],biases['b1'],biases['b2'],biases['b3'],
              biases['b4'],biases['out']]
  grads = tape.gradient(xent, var_list)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  optimizer.apply_gradients(zip(grads,var_list))

  
  if not step % 50:
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                                  tf.float32))
    losses.append(xent)
    accs.append(acc)
    print('Loss: {} Accuracy: {}'.format(xent, acc))


# In[17]:


test_layer_1 = tf.add(tf.matmul(data.test_data, weights['w1']), biases['b1'])
test_layer_2 = tf.add(tf.matmul(test_layer_1, weights['w2']), biases['b2'])
test_layer_3 = tf.add(tf.matmul(test_layer_2, weights['w3']), biases['b3'])
test_layer_4 = tf.add(tf.matmul(test_layer_3, weights['w4']), biases['b4'])
test_out = tf.matmul(test_layer_4, weights['out']) + biases['out']
test_preds = tf.argmax(test_out, axis=1, output_type=tf.int32)

test_acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),
                             tf.float32))
print('Accuracy: ')
print(int(test_acc.numpy()*100))
print("%")


# # Visulaizing loss and accuracy

# In[18]:


plt.plot(losses, 'g', label='loss')
plt.plot(accs , 'r',label='accuracy')
plt.plot(test_acc, '+' ,label = 'test_accuracy')
plt.legend(loc='upper right')
plt.show()


# In[ ]:




