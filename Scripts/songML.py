# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:45:03 2017

@author: Nguyen
"""
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

def hidden_layer(x, channels_in, channels_out,activation = None, pk = None, drop = False,name='hlayer'):
    with tf.name_scope(name):
        W = tf.Variable(tf.zeros([channels_in, channels_out]),name = 'Weights')
        b = tf.Variable(tf.zeros([channels_out]), name = 'Bias')
        if activation is 'relu':
            act = tf.nn.relu(tf.matmul(x,W) + b)
        if activation is 'sig':
            act = tf.nn.sigmoid(tf.matmul(x,W) + b)
        if activation is 'soft':
            act = tf.nn.softmax(tf.matmul(x,W) + b)
        else:
            act = tf.matmul(x, W) + b
        if drop is True:
            act = tf.nn.dropout(act, pk)
        return act



print("Reading data...")
tf.reset_default_graph()
tf.Graph().as_default()
train = pd.read_csv("../Data/train.csv")
songs = pd.read_csv("../Data/songs.csv")
#members = pd.read_csv("../Data/members.csv", parse_dates = True, infer_datetime_format = True)

song_cols = ['song_id', 'artist_name', 'genre_ids', 'language']
member_cols = ['msno', 'city', 'gender', 'registered_via']
y = train.target
OH = OneHotEncoder(sparse=False)
y = OH.fit_transform(y.values.reshape(-1,1))
train = train.merge(songs[song_cols], on = 'song_id', how='left')
#members = members.fillna(members.mean())
#train = train.merge(members[member_cols], on = 'msno', how= 'left')
train = train.fillna(0)

train = train.drop(['target'], axis = 1)
cols = list(train.columns)

for col in cols:
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        
        le = LabelEncoder()
        train_vals = list(train[col].unique())
        le.fit(train_vals)
        train[col] = le.transform(train[col])

#model = neighbors.KNeighborsClassifier(15)
#colnorm = ['msno','song_id', 'source_screen_name', 'source_type', 'artist_name']
#features = train.columns.values

#for feature in features:
#    mean, std = train[feature].mean(), train[feature].std()
#    train.loc[:,feature] = (train[feature] - mean) / std

#scaler = MinMaxScaler()
#x_scaled = scaler.fit_transform(x)

#train = pd.DataFrame(x_scaled)


print(train.head())
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)
x_train = x_train.as_matrix()
x_test = x_test.as_matrix()
print(y_train)
print("Training the model...")

## NEURAL NET ##

#PARAMS#

training_epocs = 20
training_dropout = 0.9
display_step = 1
n_samples = y_train.shape[0]
batch_size = 2046
learning_rate = 0.4

input_nodes = 8

#Multiplier for adjustment of layers

multiplier = 1.5

hidden_nodes1 = 18
hidden_nodes2 = round(hidden_nodes1 * multiplier)
hidden_nodes3 = round(hidden_nodes2 * multiplier)

#Percent of nodes to keep during dropout
percent_keep = tf.placeholder(tf.float32, name = 'percent_keep')


with tf.name_scope('Model'):
    #Inputs
    x = tf.placeholder(tf.float32, [None, input_nodes], name='Inputs')
    #Labels
    y_ = tf.placeholder(tf.float32,[None, 2], name = 'labels')
    #Hidden Layers
    l1 = hidden_layer(x, input_nodes, hidden_nodes1, activation = 'soft')
    l2 = hidden_layer(l1, hidden_nodes1, hidden_nodes2)
    l3 = hidden_layer(l2, hidden_nodes2, hidden_nodes3, drop = True, pk = percent_keep, activation = 'soft')
    lout = hidden_layer(l3, hidden_nodes3, 2, activation = 'soft')
    out = lout

with tf.name_scope('cost'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y_)
    cost = tf.reduce_mean(cross_entropy)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    #cost = cross_entropy

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
tf.summary.FileWriterCache.clear()


with tf.Session() as sess:
    
    writer = tf.summary.FileWriter('./graph10', graph = sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epocs):
        for batch in range(int(n_samples/batch_size)):
            batch_x = x_train[batch*batch_size : (1+batch) * batch_size]
            batch_y = y_train[batch*batch_size : (1+batch) * batch_size]
            sess.run([optimizer], feed_dict={x: batch_x, y_ : batch_y, percent_keep: training_dropout})
        if (epoch) % display_step == 0:
           acc, c, output = sess.run([accuracy,cost, out], feed_dict={x: x_train, y_: y_train, percent_keep: training_dropout})
           print("Epoch: ", epoch,
                 "Train Loss: ", c,
                 "Accuracy: ", acc,
                 "Output: ", output)
    writer.close()
    print("Test Accuracy: ", accuracy.eval(feed_dict={x: x_test, y_: y_test, percent_keep: training_dropout}))
    
       
print("Done")


#test_error = tf.nn.l2_loss(y_, name = "SQE")/x_test.shape[0]
#print("Test Error:", test_error.eval({x: x_test.as_matrix(), y:y_test}))

#Cost Function
#cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(out)))
#total_error = tf.reduce_sum(tf.square(tf.subtract(y_, tf.reduce_mean(y_))))
#unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_, y)))
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


