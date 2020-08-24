
import math
import tensorflow as tf
import numpy as np

from data.import_data import *

from models.networks.lstm_networks import language_generator, build_sampler
from models.networks.conv_networks import conv_decoder

class VCCA(object):

    def __init__(self, dim_x, dim_z, dim_y, dim_w, word_to_idx,
                    lambda_x=1.0, lambda_i=1.0, lambda_w=1.0,
                    n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512):
        """ VCCA for image features, iconic images, and text descriptions.

        Args:
            dim_x: Image feature dimension.
            dim_z: Latent representation dimension.
            dim_i: Iconic image dimension.
            dim_w: Text description dimension.
            word_to_idx: Vocabulary.
            lambda_x: Scaling weights for image feature view. 
            lambda_i: Scaling weights for iconic image view. 
            lambda_w: Scaling weights for text description view. 
            n_layers_encoder: Number of hidden layers in encoder. 
            n_layers_decoder: Number of hidden layers in decoder.
            fc_hidden_size: Dimension in hidden fully-connected layers 

        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.dim_w = dim_w
        # likelihood weighting terms
        self.lambda_x = lambda_x
        self.lambda_i = lambda_i
        self.lambda_w = lambda_w
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.fc_hidden_size = fc_hidden_size

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.V = len(word_to_idx) # Vocabulary size
        self.T = dim_w # Number of time steps in LSTM
        self.M = 256 # word embedding size
        self.H = dim_z # LSTM hidden state size
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')
        self.y = tf.placeholder(tf.float32, [None, dim_y[0], dim_y[1], dim_y[2]], name='y')
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1], name='captions')

        # Used batch normalization in generator
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        # Other regularizers
        self.dropout_rate = tf.placeholder_with_default(1.0, shape=(), name='dropout_rate')
        self.kl_weight = tf.placeholder_with_default(1.0, shape=(), name='kl_weight') 

        self.build_graph()
        print("Multihead VAE is initialized.")

    def build_graph(self):
        """ Build computational graph.
        """

        ### Encoder
        self.z_sample, self.z_mu, self.z_log_sigma_sq = self.get_encoder( self.x )
        self.latent_rep = self.z_mu # Latent representation
        latent_rep = tf.identity(self.latent_rep, name='latent_rep') # for fetching tensor when restoring model!

        ### Feature decoder 
        self.x_recon = self.get_decoder( self.z_sample )
        image_feature_recon = tf.identity(self.x_recon, name='image_feature_recon') 
        # Sum of squares loss
        self.x_rec_loss = tf.reduce_sum((self.x - self.x_recon)**2, 1) * self.lambda_x

        ### Iconic image decoder 
        self.y_recon = self.get_conv_decoder( self.z_sample )
        iconic_image_recon = tf.identity(self.y_recon, name='iconic_image_recon') 
        # Sum of squares loss
        self.y_rec_loss = tf.reduce_sum((self.y - self.y_recon)**2, [1, 2, 3]) * self.lambda_i

        ### Text description decoder 
        self.language_loss, self.all_h_train = self.get_text_loss( self.z_sample )
        # Categorical cross-entropy loss
        self.caption_rec_loss = tf.reduce_sum(self.language_loss, 1) * self.lambda_w

        ### KL divergence and ELBO
        self.kl_div = self.kl_divergence(self.z_mu, self.z_log_sigma_sq) * self.kl_weight   
        self.loss = -tf.reduce_mean(self.kl_div + self.x_rec_loss + self.y_rec_loss + self.caption_rec_loss)

        ### Evaluation ###
        self.z_sample_eval, _, _ = self.get_encoder( self.x, reuse = True )

        self.x_recon_eval = self.get_decoder( self.z_sample_eval, reuse = True )
        self.sampled_captions, all_hidden = self.get_text_sampler(self.z_sample, reuse=True)
        caption_sampler = tf.identity(self.sampled_captions, name='text_sampler')
        
        mask = tf.cast(tf.not_equal(tf.reduce_sum(all_hidden, axis=-1), 0), tf.float32)
        den = tf.expand_dims(tf.reduce_sum(mask, axis=-1), axis=1)
        self.text_rep = tf.reduce_sum(all_hidden, axis=1) / den 

        self.y_recon_eval = self.get_conv_decoder( self.z_sample_eval, reuse = True ) 

        ### Used in training feed_dict
        self.log_var = [self.loss, self.kl_div, self.x_rec_loss, self.y_rec_loss, self.caption_rec_loss]
        self.val_log_var = self.log_var

    def get_encoder(self, x, y=None, reuse=None):
        """ Build encoder for image features.
        """
        with tf.variable_scope('variational', reuse=reuse):

            if y is not None:
                x = tf.concat([x, y], 1)
            h = x
            for i in range(self.n_layers_encoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                h = tf.nn.leaky_relu(h)

            q_mu = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_mu')
            q_log_sigma_sq = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_log_sigma_sq')
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

    def get_conv_decoder(self, z, reuse=None):
        """ Build decoder for iconic image.
        """
        is_training = self.is_training
        with tf.variable_scope('model', reuse=reuse):
            y_recon = conv_decoder(z, is_training=is_training, reuse=reuse)
        return y_recon

    def get_text_loss(self, z, reuse=None):
        """ Build decoder for text descriptions.
        """        
        captions = self.captions
        dim_h = self.H
        word_to_idx = self.word_to_idx
        T = self.T
        dim_emb = self.M
        dropout_rate = self.dropout_rate

        with tf.variable_scope('model', reuse=reuse):
            #language_loss, h, _ = language_generator(z, captions, word_to_idx, T, dim_h, dim_emb, dropout_rate)
            language_loss, all_h = language_generator(z, captions, word_to_idx, T, dim_h, dim_emb, dropout_rate)
        return language_loss, all_h

    def get_text_sampler(self, z, reuse=True):
        """ Reusing decoder for sampling text descriptions.
        """
        dim_h = self.H
        word_to_idx = self.word_to_idx
        T = self.T
        dim_emb = self.M

        with tf.variable_scope('model', reuse=reuse):
            #generated_captions, h, _ = build_sampler(z, word_to_idx, dim_h, dim_emb, dropout_rate=1.0, max_len=16) 
            generated_captions, all_h = build_sampler(z, word_to_idx, dim_h, dim_emb, dropout_rate=1.0, max_len=16) 
        return generated_captions, all_h

    def reparameterization_trick(self, mu, log_sigma_sq):
        """
            "Reparameterization trick"
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
    captions = data['captions']
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
    w_loss = 0.
    kl_loss = 0.
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        # Prepare batch and hyperparameters 
        x_batch = features[start:end]
        y_batch = load_iconic_images(iconic_image_path[start:end])
        captions_batch = load_captions(captions, labels[start:end])
        feed_dict={model.x: x_batch, model.y: y_batch, model.captions: captions_batch,
                    model.dropout_rate: dropout_rate, model.is_training: is_training, model.kl_weight: kl_weight}

        if mode == 'train':
            # Training step
            train_step_results = session.run([train_op] + model.log_var, feed_dict=feed_dict) 
            total_loss += train_step_results[1]
            kl_loss += np.sum(train_step_results[2])
            x_loss += np.sum(train_step_results[3])
            i_loss += np.sum(train_step_results[4])
            w_loss += np.sum(train_step_results[5])

        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            kl_loss += np.sum(val_step_results[1])
            x_loss += np.sum(val_step_results[2])
            i_loss += np.sum(val_step_results[3])
            w_loss += np.sum(val_step_results[4])

        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples, 'kl_loss': kl_loss / n_examples,
                'i_loss': i_loss / n_examples, 'w_loss': w_loss / n_examples, }
    return results