
import numpy as np
import tensorflow as tf

def conv_decoder(z, is_training=False, scope_name='conv_decoder', reuse=None):
    """Get convolutional decoder for iconic images.

    Args:
        z: Input data.
        is_training: Boolean indicating to use learned parameters in batch normalization.
        scope_name: Name for the variable scope.
        reuse: Indicates if variable scopes should be reused or to create new scopes.

    Returns:
        Image in range (0, 1) and size [64, 64, 3]

    """

    nf = 64
    dim_z = int(z.get_shape().as_list()[1])

    with tf.variable_scope(scope_name, reuse=reuse):
        
        with tf.variable_scope('upconv1', reuse=reuse):
            upconv1 = tf.layers.conv2d_transpose(tf.reshape(z, (-1, 1, 1, dim_z)), filters=nf * 8, kernel_size=(4, 4), strides=(1, 1),
                            padding='valid')#, kernel_initializer=self.conv_kernel_initializer, bias_initializer=self.const_initializer)
            upconv1_bn = tf.layers.batch_normalization(upconv1, training=is_training)
            upconv1_out = tf.nn.leaky_relu(upconv1_bn) # (batch, 4, 4, nf*8)
            #print(upconv1_out.shape)

        with tf.variable_scope('upconv2', reuse=reuse):
            upconv2 = tf.layers.conv2d_transpose(upconv1_out, filters=nf * 4, kernel_size=(4, 4), strides=(2, 2),
                            padding='same')#, kernel_initializer=self.conv_kernel_initializer, bias_initializer=self.const_initializer)
            upconv2_bn = tf.layers.batch_normalization(upconv2, training=is_training)
            upconv2_out = tf.nn.leaky_relu(upconv2_bn) # (batch, 8, 8, nf*4)
            #print(upconv2_out.shape)

        with tf.variable_scope('upconv3', reuse=reuse):
            upconv3 = tf.layers.conv2d_transpose(upconv2_out, filters=nf * 2, kernel_size=(4, 4), strides=(2, 2),
                            padding='same')#, kernel_initializer=self.conv_kernel_initializer, bias_initializer=self.const_initializer )
            upconv3_bn = tf.layers.batch_normalization(upconv3, training=is_training)
            upconv3_out = tf.nn.leaky_relu(upconv3_bn) # (batch, 16, 16, nf*2)
            #print(upconv3_out.shape)

        with tf.variable_scope('upconv4', reuse=reuse):
            upconv4 = tf.layers.conv2d_transpose(upconv3_out, filters=nf, kernel_size=(4, 4), strides=(2, 2),
                            padding='same')#, kernel_initializer=self.conv_kernel_initializer, bias_initializer=self.const_initializer)
            upconv4_bn = tf.layers.batch_normalization(upconv4, training=is_training)
            upconv4_out = tf.nn.leaky_relu(upconv4_bn) # (batch, 32, 32, nf)
            #print(upconv4_out.shape)

        with tf.variable_scope('upconv5', reuse=reuse):
            upconv5 = tf.layers.conv2d_transpose(upconv4_out, filters=3, kernel_size=(4, 4), strides=(2, 2),
                            padding='same')#, kernel_initializer=self.conv_kernel_initializer, bias_initializer=self.const_initializer)
            out = tf.nn.sigmoid(upconv5) # (batch, 64, 64, 3)

    return out
    
def conv_encoder(x, out_shape=512, is_training=False, scope_name='conv_encoder', reuse=None):
    """Get convolutional decoder for iconic images.

    Args:
        x: Input data.
        out_shape: Dimension of output vector.
        is_training: Boolean indicating to use learned parameters in batch normalization.
        scope_name: Name for the variable scope.
        reuse: Indicates if variable scopes should be reused or to create new scopes.

    Returns:
        Vector with dimension out_shape with Tanh activation.

    """
    nf = 64

    with tf.variable_scope(scope_name, reuse=reuse):

        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(x, filters=nf, kernel_size=(4, 4), strides=(2, 2),
                                     padding='same', use_bias=False)
            conv1_bn = tf.layers.batch_normalization(conv1, training=is_training)
            conv1_out = tf.nn.leaky_relu(conv1_bn)

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(conv1_out, filters=nf*2, kernel_size=(4, 4), strides=(2, 2),
                                     padding='same', use_bias=False)
            conv2_bn = tf.layers.batch_normalization(conv2, training=is_training)
            conv2_out = tf.nn.leaky_relu(conv2_bn)

        with tf.variable_scope('conv3'):
            conv3 = tf.layers.conv2d(conv2_out, filters=nf*4, kernel_size=(4, 4), strides=(2, 2),
                                     padding='same', use_bias=False)
            conv3_bn = tf.layers.batch_normalization(conv3, training=is_training)
            conv3_out = tf.nn.leaky_relu(conv3_bn)

        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv2d(conv3_out, filters=nf*8, kernel_size=(4, 4), strides=(2, 2),
                                     padding='same', use_bias=False)
            conv4_bn = tf.layers.batch_normalization(conv4, training=is_training)
            conv4_out = tf.nn.leaky_relu(conv4_bn)

        # Flatten out representation
        shape = int(np.prod(conv4_out.get_shape().as_list()[1:]))
        conv_out = tf.reshape(conv4_out, [-1, shape])
        output = tf.layers.dense(conv_out, units=out_shape, activation=tf.nn.tanh, name='h1')
        
    return output