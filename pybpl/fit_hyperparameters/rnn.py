import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


def preprocess_sequences(seqs, vocab_size, max_len, one_hot_targets=True):
    # add 'start' token to head of each input sequence
    seqs_input = [[vocab_size]+s[:-1] for s in seqs]
    # add 1 to each input subID since '0' is saved for padding
    seqs_input = [np.array(s)+1 for s in seqs_input]
    # create the input and target arrays.
    X = pad_sequences(seqs_input, max_len, padding='post', truncating='post')
    Y = pad_sequences(seqs, max_len, padding='post', truncating='post')
    if one_hot_targets:
        Y = to_categorical(Y, num_classes=vocab_size)

    return X, Y

def predict(sess, model, X, batch_size=128):
    # build TF graph
    x = tf.placeholder(dtype=tf.int32, shape=(None,)+X.shape[1:])
    y_pred = model(x)

    # predict
    Y_pred = []
    nb_batches = int(np.ceil(X.shape[0] / batch_size))
    with sess.as_default():
        for i in tqdm(range(nb_batches)):
            y_pred_val = sess.run(
                y_pred,
                feed_dict={x: X[i*batch_size:(i+1)*batch_size],
                           K.learning_phase():0}
            )
            Y_pred.append(y_pred_val)
    Y_pred = np.concatenate(Y_pred)

    return Y_pred

def validate_epoch(sess, x, y, loss, X, Y, batch_size):
    nb_batches = int(np.ceil(X.shape[0] / batch_size))
    losses = []
    for i in range(nb_batches):
        X_batch = X[i*batch_size:(i+1)*batch_size]
        Y_batch = Y[i*batch_size:(i+1)*batch_size]
        loss_val = sess.run(
            loss,
            feed_dict={x: X_batch, y: Y_batch, K.learning_phase(): 0}
        )
        losses.append(loss_val)

    return np.mean(losses)

def train_epoch(sess, x, y, loss, train_op, X, Y, batch_size):
    nb_batches = int(np.ceil(X.shape[0] / batch_size))
    losses = []
    for i in tqdm(range(nb_batches)):
        X_batch = X[i*batch_size:(i+1)*batch_size]
        Y_batch = Y[i*batch_size:(i+1)*batch_size]
        loss_val, _ = sess.run(
            [loss, train_op],
            feed_dict={x: X_batch, y: Y_batch, K.learning_phase(): 1}
        )
        losses.append(loss_val)

    return np.mean(losses)

def train(sess, model, X, Y, loss_fn, epochs, validation_split=0.2,
          batch_size=32, save_file=None):
    x = tf.placeholder(dtype=tf.int32, shape=(None,)+X.shape[1:])
    if Y.dtype.kind == 'i':
        y = tf.placeholder(dtype=tf.int32, shape=(None,)+Y.shape[1:])
    elif Y.dtype.kind == 'f':
        y = tf.placeholder(dtype=tf.float32, shape=(None,)+Y.shape[1:])
    else:
        raise Exception
    y_pred = model(x)
    loss = loss_fn(x, y, y_pred)

    # create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, var_list=model.trainable_weights)

    # train-valid split
    assert X.shape[0] == Y.shape[0]
    assert validation_split > 0.01
    nb_train = int(X.shape[0]*(1.-validation_split))
    X_train, Y_train = X[:nb_train], Y[:nb_train]
    X_valid, Y_valid = X[nb_train:], Y[nb_train:]

    # train the model
    train_losses = []
    valid_losses = []
    ix = np.arange(X_train.shape[0]) # training indices. These will be shuffled
    with sess.as_default() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            # shuffle training indices
            np.random.shuffle(ix)
            X_train, Y_train = X_train[ix], Y_train[ix]
            print('Epoch # %i/%i' % (epoch+1, epochs))
            time.sleep(0.2)
            train_loss = train_epoch(sess, x, y, loss, train_op, X_train, Y_train, batch_size)
            time.sleep(0.2)
            valid_loss = validate_epoch(sess, x, y, loss, X_valid, Y_valid, batch_size)
            print('train_loss: %0.4f - valid_loss: %0.4f\n' % (train_loss, valid_loss))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if valid_loss <= np.min(valid_losses):
                model.save(save_file)

    return model, train_losses, valid_losses