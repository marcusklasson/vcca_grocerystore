
import math
import tensorflow as tf
import numpy as np

from data.import_data import load_iconic_images

from models.networks.conv_networks import conv_encoder, conv_decoder

class VCCA_private(object):

    def __init__(self, dim_x, dim_z, dim_i, lambda_x=1.0, lambda_i=1.0,
                     n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512):
        """ VCCA_private for image features and iconic images.

        Args:
            dim_x: Image feature dimension.
            dim_z: Latent representation dimension.
            dim_i: Iconic image dimension.
            lambda_x: Scaling weights for image feature view. 
            lambda_i: Scaling weights for iconic image view. 
            n_layers_encoder: Number of hidden layers in encoder. 
            n_layers_decoder: Number of hidden layers in decoder.
            fc_hidden_size: Dimension in hidden fully-connected layers 

        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_i = dim_i

        self.dim_ux = dim_z
        self.dim_ui = dim_z 
        # likelihood weighting terms
        self.lambda_x = lambda_x
        self.lambda_i = lambda_i
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.fc_hidden_size = fc_hidden_size
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')
        self.iconic_images = tf.placeholder(tf.float32, [None, dim_i[0], dim_i[1], dim_i[2]], name='iconic_images')

        # Used batch normalization in generator
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        # Other regularizers
        self.kl_weight = tf.placeholder_with_default(1.0, shape=(), name='kl_weight') 

        self.build_graph()
        print("VCCA_private_xi is initialized.")

    def build_graph(self):
        """ Build computational graph.
        """

        ### Encoder for shared latent variable z
        self.z_sample, self.z_mu, self.z_log_sigma_sq = self.get_encoder(self.x, name='variational/z')
        self.latent_rep = self.z_mu # Latent representation
        latent_rep = tf.identity(self.latent_rep, name='latent_rep') # for fetching tensor when restoring model!

        # Encoder for image feature private latent hx
        self.ux_sample, self.ux_mu, self.ux_log_sigma_sq = self.get_encoder(self.x, name='variational/ux')
        self.latent_rep_ux = self.ux_mu # Latent representation for natural image view
        ux_mu = tf.identity(self.ux_mu, name='latent_rep_ux') # for fetching tensor when restoring model!

        ### Iconic image encoder
        self.ui_sample, self.ui_mu, self.ui_log_sigma_sq = self.get_conv_encoder(self.iconic_images, name='variational/ui')
        self.latent_rep_ui = self.ui_mu # Latent representation for iconic image view
        ui_mu = tf.identity(self.ui_mu, name='latent_rep_ui') # for fetching tensor when restoring model!

        ### Feature decoder 
        self.x_recon = self.get_decoder(self.z_sample, y=self.ux_sample)
        image_feature_recon = tf.identity(self.x_recon, name='image_feature_recon') 
        # Sum of squares loss
        self.x_rec_loss = tf.reduce_sum((self.x - self.x_recon)**2, 1) * self.lambda_x

        ### Iconic image decoder 
        self.i_recon = self.get_conv_decoder(self.z_sample, y=self.ui_sample)
        iconic_image_recon = tf.identity(self.i_recon, name='iconic_image_recon') 
        # Sum of squares loss
        self.i_rec_loss = tf.reduce_sum((self.iconic_images - self.i_recon)**2, [1, 2, 3]) * self.lambda_i

        ### KL divergence and ELBO
        self.kl_div_z = self.kl_divergence(self.z_mu, self.z_log_sigma_sq) * self.kl_weight
        self.kl_div_ux = self.kl_divergence(self.ux_mu, self.ux_log_sigma_sq) 
        self.kl_div_ui = self.kl_divergence(self.ui_mu, self.ui_log_sigma_sq)    
        self.loss = -tf.reduce_mean(self.kl_div_z + self.x_rec_loss + self.i_rec_loss + self.kl_div_ux + self.kl_div_ui)

        ### Evaluation ###
        self.z_sample_eval, _, _ = self.get_encoder(self.x, name='variational/z', reuse=True)
        self.ux_sample_eval, _, _ = self.get_encoder(self.x, name='variational/ux', reuse=True)
        self.ui_sample_eval, _, _ = self.get_conv_encoder(self.iconic_images, name='variational/ui', reuse=True)
        self.x_recon_eval = self.get_decoder(self.z_sample_eval, y=self.ux_sample_eval, reuse=True)
        self.i_recon_eval = self.get_conv_decoder(self.z_sample_eval, y=self.ui_sample_eval, reuse=True) 

        ### Used in training feed_dict
        self.log_var = [self.loss, self.kl_div_z, self.x_rec_loss, self.i_rec_loss, self.kl_div_ux, self.kl_div_ui]
        self.val_log_var = self.log_var

    def get_encoder(self, x, y=None, name='variational', reuse=None):
        """ Build encoder for image features.
        """
        with tf.variable_scope(name, reuse=reuse):

            if y is not None:
                x = tf.concat([x, y], 1)
            h = x
            for i in range(self.n_layers_encoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                h = tf.nn.leaky_relu(h)

            q_mu = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_mu')
            q_log_sigma_sq = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_logvar')
            z = self.reparameterization_trick(q_mu, q_log_sigma_sq)
        return z, q_mu, q_log_sigma_sq  

    def get_decoder(self, z, y=None, reuse=None):
        """ Build decoder for image features.
        """
        with tf.variable_scope('model', reuse=reuse):

            if y is not None:
                z = tf.concat([z, y], 1)
            h = z
            for i in range(self.n_layers_decoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                h = tf.nn.leaky_relu(h)
            x_recon = tf.layers.dense(h, units=self.dim_x, activation=None, name='x_recon')
        return x_recon

    def get_conv_decoder(self, z, y=None, reuse=None):
        """ Build decoder for iconic image.
        """
        is_training = self.is_training
        if y is not None:
            z = tf.concat([z, y], 1)
        with tf.variable_scope('model', reuse=reuse):
            y_recon = conv_decoder(z, is_training=is_training, reuse=reuse)
        return y_recon

    def get_conv_encoder(self, x, name='variational', reuse=None):
        """ Build encoder for iconic image.
        """
        is_training = self.is_training
        with tf.variable_scope(name, reuse=reuse):
            h = conv_encoder(x, is_training=is_training, reuse=reuse)
            q_mu = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_mu_i')
            q_logvar = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_logvar_i')
            z = self.reparameterization_trick(q_mu, q_logvar)
        return z, q_mu, q_logvar

    def reparameterization_trick(self, mu, log_sigma_sq):
        """ Reparameterization trick
            
            z = mu + sigma * epsilon
        """
        epsilon = tf.random_normal((tf.shape(mu)), 0., 1. )
        sample = mu + tf.exp(0.5*log_sigma_sq) * epsilon
        return sample

    def kl_divergence(self, mu, log_sigma_sq):
        """ KL divergence for two Gaussians.
        """
        return -0.5*tf.reduce_sum(1 + log_sigma_sq - mu**2 - tf.exp(log_sigma_sq), 1)  

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
    dropout_rate = hyperparams['dropout_rate']
    kl_weight = hyperparams['kl_weight']
    is_training = hyperparams['is_training']

    # Data
    features = data['features']
    labels = data['labels']
    iconic_image_path = data['iconic_image_paths']
    n_classes = data['n_classes']

    n_examples = len(features)
    n_batches = int(np.ceil(n_examples/batch_size))

    if shuffle:
        perm = np.random.permutation(n_examples)
        features = features[perm]
        iconic_image_path = iconic_image_path[perm]
        labels = labels[perm]
        
    total_loss = 0.
    x_loss = 0.
    i_loss = 0.
    kl_loss_z = 0.
    kl_loss_ux = 0.
    kl_loss_ui = 0.
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        # Prepare batch and hyperparameters 
        x_batch = features[start:end]
        i_batch = load_iconic_images(iconic_image_path[start:end])
        feed_dict={model.x: x_batch, model.iconic_images: i_batch,
                    model.kl_weight: kl_weight, model.is_training: is_training}

        if mode == 'train':
            # Training step
            train_step_results = session.run([train_op] + model.log_var, feed_dict=feed_dict) 
            total_loss += train_step_results[1]
            kl_loss_z += np.sum(train_step_results[2])
            x_loss += np.sum(train_step_results[3])
            i_loss += np.sum(train_step_results[4])
            kl_loss_ux += np.sum(train_step_results[5])
            kl_loss_ui += np.sum(train_step_results[6])

        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            kl_loss_z += np.sum(val_step_results[1])
            x_loss += np.sum(val_step_results[2])
            i_loss += np.sum(val_step_results[3])
            kl_loss_ux += np.sum(val_step_results[4])
            kl_loss_ui += np.sum(val_step_results[5])

        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples, 'i_loss': i_loss / n_examples,
                'kl_loss_z': kl_loss_z / n_examples, 'kl_loss_ux': kl_loss_ux / n_examples, 'kl_loss_ui': kl_loss_ui / n_examples,  }
    return results