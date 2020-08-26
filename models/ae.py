
import tensorflow as tf
import numpy as np

class Autoencoder(object):

    def __init__(self, dim_x, dim_z, n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512, lambda_x=1.0,
        use_batchnorm=False):
        """ Autoencoder for compressing image features.

        Args:
            dim_x: Image feature dimension.
            dim_z: Latent representation dimension.
            n_layers_encoder: Number of hidden layers in encoder. 
            n_layers_decoder: Number of hidden layers in decoder.
            fc_hidden_size: Dimension in hidden fully-connected layers 
            lambda_x: Scaling weights for image feature view. 
            use_batchnorm: Use batch normalizatin in network or not.

        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.lambda_x = lambda_x 
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.fc_hidden_size = fc_hidden_size

        self.use_batchnorm = use_batchnorm
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')

        self.build_graph()
        print("Autoencoder is initialized.")

    def build_graph(self):
        """ Build computational graph.
        """

        ### Encoder
        self.z_emb = self.get_encoder(self.x)
        self.latent_rep = self.z_emb # Latent representation
        latent_rep = tf.identity(self.latent_rep, name='latent_rep') # for fetching tensor when restoring model!

        ### Feature decoder 
        self.x_recon = self.get_decoder( self.z_emb )
        image_feature_recon = tf.identity(self.x_recon, name='image_feature_recon') 
        # Sum of squares loss
        self.x_rec_loss = tf.reduce_sum((self.x - self.x_recon)**2, 1) * self.lambda_x

        ### Reconstruction loss  
        self.loss = -tf.reduce_mean(self.x_rec_loss)

        ### Evaluation ###
        self.z_emb_eval = self.get_encoder( self.x, reuse = True )

        self.x_recon_eval = self.get_decoder( self.z_emb_eval, reuse = True )

        ### Used in training feed_dict
        self.log_var = [self.loss, self.x_rec_loss]
        self.val_log_var = self.log_var

    def get_encoder(self, x, y=None, reuse=None):
        with tf.variable_scope('encoder', reuse=reuse):

            if y is not None:
                x = tf.concat([x, y], 1)
            h = x
            for i in range(self.n_layers_encoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                if self.use_batchnorm:
                    h = tf.layers.batch_normalization(h, training=self.is_training, name='bn_' + str(i+1))
                h = tf.nn.leaky_relu(h)
            z = tf.layers.dense(h, units=self.dim_z, activation=None, name='z')
        return z

    def get_decoder(self, z, y=None, reuse=None):
        with tf.variable_scope('model', reuse=reuse):

            if y is not None:
                z = tf.concat([z, y], 1)
            h = z
            for i in range(self.n_layers_decoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                if self.use_batchnorm:
                    h = tf.layers.batch_normalization(h, training=self.is_training, name='bn_' + str(i+1))
                h = tf.nn.leaky_relu(h)
            x_recon = tf.layers.dense(h, units=self.dim_x, activation=None, name='x_recon')
        return x_recon

def run_training_epoch(args, data, model, hyperparams, session, train_op=None, shuffle=False, mode='train'):
    """ Execute training epoch for Autoencoder.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        data: Data used during epoch.
        model: Model used during epoch.
        hyperparams: Hyperparameters for training.
        session: Tensorflow session. 
        train_op: Op for computing gradients and updating parameters in model.
        shuffle: For shuffling data before epoch.
        mode: Training/validation mode.

    Returns:
        Metrics in python dictionary.

    """

    # Hyperparameters
    batch_size = hyperparams['batch_size']
    kl_weight = hyperparams['kl_weight']
    is_training = hyperparams['is_training']

    # Data
    features = data['features']
    n_examples = len(features)
    n_batches = int(np.ceil(n_examples/batch_size))
    if shuffle:
        perm = np.random.permutation(n_examples)
        features = features[perm]
        
    total_loss = 0.
    x_loss = 0.
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        # Prepare batch and hyperparameters 
        x_batch = features[start:end]
        feed_dict={model.x: x_batch}

        if mode == 'train':
            # Training step
            train_step_results = session.run([train_op] + model.log_var, feed_dict=feed_dict) 
            total_loss += train_step_results[1]
            x_loss += np.sum(train_step_results[2])

        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            x_loss += np.sum(val_step_results[1])

        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples}
    return results