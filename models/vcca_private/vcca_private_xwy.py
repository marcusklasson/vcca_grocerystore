
import tensorflow as tf
import numpy as np

from data.import_data import load_captions
from data.data_processing import onehot_encode

from models.networks.lstm_networks import language_generator, build_sampler, language_encoder

class VCCA_private(object):

    def __init__(self, dim_x, dim_z, dim_labels, dim_w, word_to_idx,
                     n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512,
                     n_layers_classifier=1, lambda_x=1.0, lambda_w=1.0, lambda_y=1.0):
        """ VCCA_private for image features, text descriptions, and labels.

        Args:
            dim_x: Image feature dimension.
            dim_z: Latent representation dimension.
            dim_labels: Label dimension (number of classes).
            dim_w: Text description dimension.
            word_to_idx: Vocabulary.
            lambda_x: Scaling weights for image feature view. 
            lambda_w: Scaling weights for text description view. 
            lambda_y: Scaling weights for class label view. 
            n_layers_encoder: Number of hidden layers in encoder. 
            n_layers_decoder: Number of hidden layers in decoder.
            fc_hidden_size: Dimension in hidden fully-connected layers 
            n_layers_classifier: Number of hidden layers in class label decoder.

        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_labels = dim_labels
        self.dim_w = dim_w
        self.dim_ux = dim_z
        self.dim_uw = dim_z 

        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.lambda_y = lambda_y
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_classifier = n_layers_classifier
        self.fc_hidden_size = fc_hidden_size

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.V = len(word_to_idx) # Vocabulary size
        self.T = dim_w # Number of time steps in LSTM
        self.M = 256 # word embedding size
        self.H = dim_z # LSTM hidden state size
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1], name='captions')
        self.labels = tf.placeholder(tf.float32, [None, self.dim_labels], name='labels')

        # Other regularizers
        self.dropout_rate = tf.placeholder_with_default(1.0, shape=(), name='dropout_rate')
        self.kl_weight = tf.placeholder_with_default(1.0, shape=(), name='kl_weight')
        self.K = tf.placeholder_with_default(1, shape=(), name='posterior_samples')  

        self.build_graph()
        print("VCCA_private_xwy is initialized.")

    def build_graph(self):
        """ Build computational graph
        """

        ### Encoder for shared latent variable z
        self.z_sample, self.z_mu, self.z_log_sigma_sq = self.get_encoder(self.x, name='variational/z')
        self.latent_rep = self.z_mu # Latent representation
        latent_rep = tf.identity(self.latent_rep, name='latent_rep_z') # for fetching tensor when restoring model!

        # Encoder for image feature private latent hx
        self.ux_sample, self.ux_mu, self.ux_log_sigma_sq = self.get_encoder(self.x, name='variational/ux')
        self.latent_rep_ux = self.ux_mu # Latent representation for natural_image view
        latent_rep_ux = tf.identity(self.ux_mu, name='latent_rep_ux') # for fetching tensor when restoring model!

         # Encoder for image feature private latent hx
        self.uw_sample, self.uw_mu, self.uw_log_sigma_sq = self.get_text_encoder(self.captions,
                                                                                 dim_z=self.dim_uw, 
                                                                                 name='variational/uw')
        self.latent_rep_uw = self.uw_mu # Latent representation for text description view
        latent_rep_uw = tf.identity(self.uw_mu, name='latent_rep_uw') # for fetching tensor when restoring model!

        ### Feature decoder 
        self.x_recon = self.get_decoder(self.z_sample, y=self.ux_sample)
        image_feature_recon = tf.identity(self.x_recon, name='image_feature_recon') 
        # Sum of squares loss
        self.x_rec_loss = tf.reduce_sum((self.x - self.x_recon)**2, 1) * self.lambda_x

        ### Text description decoder 
        self.language_loss, self.all_h_train = self.get_text_loss(self.z_sample, y=self.uw_sample)
        # Categorical cross-entropy loss
        self.w_rec_loss = tf.reduce_sum(self.language_loss, 1) * self.lambda_w

        # Classifier
        self.logits = self.get_classifier(self.z_sample)
        self.clf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits) * self.lambda_y

        self.correct_pred = tf.equal(tf.argmax(self.labels, axis=-1), tf.argmax(self.logits, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        ### KL divergence and ELBO
        self.kl_div_z = self.kl_divergence(self.z_mu, self.z_log_sigma_sq) * self.kl_weight   
        self.kl_div_ux = self.kl_divergence(self.ux_mu, self.ux_log_sigma_sq) 
        self.kl_div_uw = self.kl_divergence(self.uw_mu, self.uw_log_sigma_sq) 
        self.loss = -tf.reduce_mean(self.kl_div_z + self.x_rec_loss + self.w_rec_loss + 
                                    self.clf_loss + self.kl_div_ux + self.kl_div_uw)

        ### Used in training feed_dict
        self.log_var = [self.loss, self.kl_div_z, self.x_rec_loss, self.w_rec_loss,
                         self.clf_loss, self.accuracy, self.kl_div_ux, self.kl_div_uw]

        ### Evaluation ###
        self.z_sample_eval, _, _ = self.get_encoder(self.x, name='variational/z', reuse = True)
        self.ux_sample_eval, _, _ = self.get_encoder(self.x, name='variational/ux', reuse = True)
        self.uw_sample_eval, _, _ = self.get_text_encoder(self.captions, dim_z=self.dim_uw, 
                                                            name='variational/uw', reuse = True)

        self.x_recon_eval = self.get_decoder( self.z_sample_eval, y=self.ux_sample_eval, reuse = True )
        self.sampled_captions, all_hidden = self.get_text_sampler(self.z_sample, y=self.uw_sample_eval, reuse=True)
        caption_sampler = tf.identity(self.sampled_captions, name='text_sampler')
        
        mask = tf.cast(tf.not_equal(tf.reduce_sum(all_hidden, axis=-1), 0), tf.float32)
        den = tf.expand_dims(tf.reduce_sum(mask, axis=-1), axis=1)
        self.text_rep = tf.reduce_sum(all_hidden, axis=1) / den 

        self.val_logits = self.get_classifier(self.z_sample_eval, reuse=True)
        self.val_score = tf.cond(self.K > 1, lambda: self.compute_prediction_score(self.val_logits, mode='avg'),
                                             lambda: tf.nn.softmax(self.val_logits))
        clf_val_score = tf.identity(self.val_score, name='classifier_val_softmax_score')

        self.val_correct_pred = tf.equal(tf.argmax(self.labels, axis=-1), tf.argmax(self.val_score, axis=-1))
        self.val_accuracy = tf.reduce_mean(tf.cast(self.val_correct_pred, tf.float32))
        clf_val_accuracy = tf.identity(self.val_accuracy, name='classifier_val_accuracy')

        self.val_log_var = [self.loss, self.kl_div_z, self.x_rec_loss, self.w_rec_loss,
                             self.clf_loss, self.val_accuracy, self.kl_div_ux, self.kl_div_uw]

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
            q_logvar = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_logvar')
            z_sample, q_mu, q_logvar = self.reparameterization_trick(q_mu, q_logvar)
        return z_sample, q_mu, q_logvar  

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

    def get_text_encoder(self, w, dim_z, name='variational', reuse=None):
        """ Build encoder for text desxcriptions.
        """
        with tf.variable_scope(name, reuse=reuse):
            dim_h = self.H
            word_to_idx = self.word_to_idx
            T = self.T
            dim_emb = self.M

            self.all_hidden, self.final_hidden = language_encoder(w, word_to_idx, T, dim_h, dim_emb, reuse=None)

            #h = self.final_hidden # use final hidden states
            h = self._avg_hidden_states(self.all_hidden) #use average of hidden states
            q_mu = tf.layers.dense(h, units=dim_z, activation=None, name='q_mu')
            q_logvar = tf.layers.dense(h, units=dim_z, activation=None, name='q_logvar')
            z_sample, q_mu, q_logvar  = self.reparameterization_trick(q_mu, q_logvar)
        return z_sample, q_mu, q_logvar

    def get_text_loss(self, z, y=None, reuse=None):
        """ Build decoder for text descriptions.
        """
        captions = self.captions
        dim_h = self.H
        word_to_idx = self.word_to_idx
        T = self.T
        dim_emb = self.M
        dropout_rate = self.dropout_rate

        with tf.variable_scope('model', reuse=reuse):
            if y is not None:
                z = tf.concat([z, y], 1)
            #language_loss, h, _ = language_generator(z, captions, word_to_idx, T, dim_h, dim_emb, dropout_rate)
            language_loss, all_h = language_generator(z, captions, word_to_idx, T, dim_h, dim_emb, dropout_rate)
        return language_loss, all_h

    def get_text_sampler(self, z, y=None, reuse=True):
        """ Reusing decoder for sampling text descriptions.
        """
        dim_h = self.H
        word_to_idx = self.word_to_idx
        T = self.T
        dim_emb = self.M

        with tf.variable_scope('model', reuse=reuse):
            if y is not None:
                z = tf.concat([z, y], 1)
            #generated_captions, h, _ = build_sampler(z, word_to_idx, dim_h, dim_emb, dropout_rate=1.0, max_len=16) 
            generated_captions, all_h = build_sampler(z, word_to_idx, dim_h, dim_emb, dropout_rate=1.0, max_len=16) 
        return generated_captions, all_h

    def get_classifier(self, z, reuse=None):
        """ Build class label decoder.
        """
        dim_input = self.dim_z
        with tf.variable_scope('classifier', reuse=reuse) as scope:
            h = z
            for i in range(self.n_layers_classifier):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                h = tf.nn.leaky_relu(h)
            y_logits = tf.layers.dense(h, units=self.dim_labels, activation=None, name='y_logits')
        return y_logits

    def compute_prediction_score(self, logits, mode='avg'):
        """ Compute prediction scores when number of 
            posterior samples K > 1 in class label decoder.
        """
        N = tf.shape(self.x)[0]
        score = tf.nn.softmax(logits)
        score = tf.reshape(score, [self.K, N, self.dim_labels])
        score = tf.transpose(score, [1, 2, 0]) # y_score.shape = [N, y_dim, K]
        if mode == 'avg':
            score = tf.reduce_mean(score, 2)
        if mode == 'max_vote':
            vote = tf.equal(tf.reduce_max(score, axis=1, keepdims=True), score) 
            vote = tf.cast(vote, tf.float32)
            print('y_vote.shape: ', vote.shape)
            score = tf.reduce_sum(vote, 2)
        return score

    def reparameterization_trick(self, mu, log_sigma_sq):
        """
            "Reparameterization trick"
            z = mu + sigma * epsilon
        """
        K = self.K
        mu = tf.tile(mu, [K, 1])
        log_sigma_sq = tf.tile(log_sigma_sq, [K, 1])
        sample = mu + tf.exp(0.5*log_sigma_sq) * tf.random_normal((tf.shape(mu)), 0., 1. )
        return sample, mu, log_sigma_sq

    def _avg_hidden_states(self, all_h):
        """ Averaging hidden states to get representation for text description.
        """
        mask = tf.cast(tf.not_equal(tf.reduce_sum(all_h, axis=-1), 0), tf.float32)
        den = tf.expand_dims(tf.reduce_sum(mask, axis=-1), axis=1)
        text_rep = tf.reduce_sum(all_h, axis=1) / den 
        return text_rep

    def kl_divergence(self, mu, log_sigma_sq):
        """ KL divergence between two Gaussians.
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
    n_classes = data['n_classes']

    n_examples = len(features)
    n_batches = int(np.ceil(n_examples/batch_size))
    if shuffle:
        perm = np.random.permutation(n_examples)
        features = features[perm]
        labels = labels[perm]
        
    total_loss = 0.
    x_loss = 0.
    w_loss = 0.
    clf_loss = 0.
    accuracy = 0.
    kl_loss_z = 0.
    kl_loss_ux = 0.
    kl_loss_uw = 0.
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        # Prepare batch and hyperparameters 
        x_batch = features[start:end]
        labels_batch = onehot_encode(labels[start:end], n_classes)
        captions_batch = load_captions(captions, labels[start:end])
        feed_dict={model.x: x_batch, model.labels: labels_batch,
                    model.captions: captions_batch, model.dropout_rate: dropout_rate}

        if mode == 'train':
            # Training step
            train_step_results = session.run([train_op] + model.log_var, feed_dict=feed_dict) 
            total_loss += train_step_results[1]
            kl_loss_z += np.sum(train_step_results[2])
            x_loss += np.sum(train_step_results[3])
            w_loss += np.sum(train_step_results[4])
            clf_loss += np.sum(train_step_results[5])
            accuracy += np.sum(train_step_results[6])
            kl_loss_ux += np.sum(train_step_results[7])
            kl_loss_uw += np.sum(train_step_results[8])

        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            kl_loss_z += np.sum(val_step_results[1])
            x_loss += np.sum(val_step_results[2])
            w_loss += np.sum(val_step_results[3])
            clf_loss += np.sum(val_step_results[4])
            accuracy += np.sum(val_step_results[5])
            kl_loss_ux += np.sum(val_step_results[6])
            kl_loss_uw += np.sum(val_step_results[7])

        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples, 'w_loss': w_loss / n_examples, 
                'kl_loss_z': kl_loss_z / n_examples, 'kl_loss_ux': kl_loss_ux / n_examples, 'kl_loss_uw': kl_loss_uw / n_examples, 
                'clf_loss': clf_loss / n_examples, 'accuracy': accuracy / n_batches}
    return results