# Song_Prediciton

This program is used to demonstrate the use of a multi-layer perceptron used on a song prediciton dataset found at kaggle.com ( https://www.kaggle.com/c/kkbox-music-recommendation-challenge).
The purpose of this is to demonstrate the use of tensorflow to create a feed forward network that creates a straightforward graph for new users.

## Getting Started

This scirpt uses python 3.5 and tensorflow 1.1.0
Supporting libraries are :
pandas for data structuring
sklearn for model accuracy and splitting our train and test set

## Usage

I reccomend using a python virtual environment specifically for use of tensorflow

```python
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
        if activation is 'tanh':
            act = tf.nn.tanh(tf.matmul(x,W) + b)
        else:
            act = tf.matmul(x, W) + b
        if drop is True:
            act = tf.nn.dropout(act, pk)
        return act
```

Is the basis of building our network, it is a simple funciton I put together to allow you to build easy MLP layers.

x: Input

channels_in/out: The amount of tensors going and and out of a node respectively

activation: This can be relu,sigmoid,softmax or the hyperbolic tangent

pk: Percent keep if dropout is used on a layer

drop: Weather or not to use layer dropout


## Graph Creation

### Full model:
![](https://i.imgur.com/XVycuiC.png)


### Layers opened:

![](https://imgur.com/w1kBDbG.png)

Each run builds a clean tensorboard graph that is used to demonstrate the framework of the network
it will be put into a folder graph12 and can be run from command line by:

> activate tensorflow virtual environment

tensorboard --logdir graph12

From here you can go to your localhost:6006 and under the "graphs" section you may view your interactive board

## Author

Jacob Biloki : bilokij@gmail.com

## License

This project is licensed under the MIT License
