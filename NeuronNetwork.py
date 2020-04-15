import tensorflow as tf
import tensorflow.compat.v1 as tfc
import matplotlib.pyplot as plt
import tensorflow_core.examples.tutorials.mnist.input_data as input_data

def main():
    tf.compat.v1.disable_eager_execution()
    # data = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = data.load_data()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # plt.imshow(x_train[1], cmap="binary")
    # plt.show()
    sess = tfc.InteractiveSession()
    x = tfc.placeholder("float", shape=[None, 784])
    y_ = tfc.placeholder("float", shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tfc.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tfc.log(y))
    train_step = tfc.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("测试集的正确率为",accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main()
