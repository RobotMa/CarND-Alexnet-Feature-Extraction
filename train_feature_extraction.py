import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

nb_classes = 43
epochs = 13
batch_size = 128
rate = 0.001

# TODO: Load traffic signs data.
total_file = 'train.p'
with open(total_file, mode='rb') as f:
    total = pickle.load(f)
X_total, y_total = total['features'], total['labels']

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split( \
        X_total, y_total, test_size = 0.3, random_state = 42)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
one_hot_y = tf.one_hot(labels, nb_classes)
resized = tf.image.resize_images(features, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(shape[1]))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        batch_x, batch_y = X[offset:end], y[offset:end]
        loss, acc = sess.run([loss_operation, accuracy_operation], feed_dict={features:batch_x, labels:y_batch})
        total_loss += (loss*batch_x.shape[0])
        total_acc  += (acc*batch_x.shape[0])
    return total_loss/X.shape[0], total_acc/X.shape[0]

# TODO: Train and evaluate the feature extraction model.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('training ...')
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={features:batch_x, labels:batch_y})

        val_loss, val_acc = eval_on_data(X_valid, y_valid, sess)

        print("Epoch", i+1)
        print("Time: {:.3f} seconds".format(time.time() - t0))
        print("Validation Loss = {:.3f}".format(val_acc))
        print("Validation Accuracy = {:.3f}".format(val_loss))

