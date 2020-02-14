import tensorflow as tf
import input_data
import numpy as np
from PIL import Image

batch_size = 64
z_dim = 100
learning_rate = 0.001
beta1 = 0.5
epochs = 5000


def model_inputs(image_width, image_height, image_channels, z_dim):
    # Real imag
    inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')

    # input z

    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    # Learning rate
    learning_rate = tf.placeholder(tf.float32, name='lr')

    return inputs_real, inputs_z, learning_rate


def generator(input_z, out_channel_dim, is_train=True):
    with tf.variable_scope('generator', reuse=not is_train):
        x0 = tf.layers.dense(input_z, 4 * 4 * 512)
        x0 = tf.reshape(x0, (-1, 4, 4, 512))
        bn0 = tf.layers.batch_normalization(x0, training=is_train)
        relu0 = tf.nn.relu(bn0)

        # 反卷积
        x1 = tf.layers.conv2d_transpose(relu0, 256, 4, strides=1, padding='valid')
        bn1 = tf.layers.batch_normalization(x1, training=is_train)
        relu1 = tf.nn.relu(bn1)

        x2 = tf.layers.conv2d_transpose(relu1, 512, 3, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=is_train)
        relu2 = tf.nn.relu(bn2)

        logits = tf.layers.conv2d_transpose(relu2, out_channel_dim, 3, strides=2, padding='same')
        out = tf.tanh(logits)

    return out


def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param images: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    # TODO: Implement Function

    # scope here

    with tf.variable_scope('discriminator', reuse=reuse):
        alpha = 0.2  # leak relu coeff

        # drop out probability
        keep_prob = 0.8

        # input layer 28 * 28 * color channel
        x1 = tf.layers.conv2d(images, 128, 5, strides=2, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
        # No batch norm here
        # leak relu here / alpha = 0.2
        relu1 = tf.maximum(alpha * x1, x1)
        # applied drop out here
        drop1 = tf.nn.dropout(relu1, keep_prob=keep_prob)
        # 14 * 14 * 128

        # Layer 2
        x2 = tf.layers.conv2d(drop1, 256, 5, strides=2, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
        # employ batch norm here
        bn2 = tf.layers.batch_normalization(x2, training=True)
        # leak relu
        relu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = tf.nn.dropout(relu2, keep_prob=keep_prob)

        # 7 * 7 * 256

        # Layer3
        x3 = tf.layers.conv2d(drop2, 512, 5, strides=2, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        drop3 = tf.nn.dropout(relu3, keep_prob=keep_prob)
        # 4 * 4 * 512

        # Output
        # Flatten
        flatten = tf.reshape(relu3, (-1, 4 * 4 * 512))
        logits = tf.layers.dense(flatten, 1)
        # activation
        out = tf.nn.sigmoid(logits)

    return out, logits


def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function

    g_model = generator(input_z, out_channel_dim, is_train=True)

    g_model1 = generator(input_z, out_channel_dim, is_train=False)

    d_model_real, d_logits_real = discriminator(input_real, reuse=False)

    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    ## add smooth here

    smooth = 0.1
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * (1 - smooth)))

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss, g_model1


def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, data_shape):
    input_real, input_z, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss, g_out = model_loss(input_real, input_z, data_shape[-1])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    steps = 0
    losses = []
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            steps += 1
            batch = mnist.train.next_batch(batch_size)
            labels = batch[1]
            batch = batch[0]
            train_batch = []
            for index, label in enumerate(labels):
                if label[2] == 1:
                    train_batch.append(batch[index])
            input_batch_size = len(train_batch)
            train_batch = np.array(train_batch)
            train_batch = np.reshape(train_batch, (-1, data_shape[1], data_shape[2], data_shape[3]))
            batch_z = np.random.uniform(-1, 1, size=(input_batch_size, z_dim))
            _ = sess.run(d_opt, feed_dict={input_real: train_batch, input_z: batch_z, lr: learning_rate})
            _ = sess.run(g_opt, feed_dict={input_real: train_batch, input_z: batch_z, lr: learning_rate})
            if steps % 10 == 0:
                train_loss_d = d_loss.eval({input_real: train_batch, input_z: batch_z})
                train_loss_g = g_loss.eval({input_real: train_batch, input_z: batch_z})

                losses.append((train_loss_d, train_loss_g))

                print("Epoch {}/{}...".format(epoch_i + 1, epochs),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "Generator Loss: {:.4f}".format(train_loss_g))
            if steps % 100 == 0:
                img = g_out.eval({input_z: batch_z})
                img = img[-1] * 128 + 128
                img = img.astype(int)
                img = img[:, :, 0]
                im = Image.fromarray(img).convert('L')
                im.save("result_{}.png".format(steps))
                # im.show()


if __name__ == "__main__":
    train(epochs, batch_size, z_dim, learning_rate, beta1, [0, 28, 28, 1])
