
import tensorflow as tf
import numpy as np

from models.networks.lstm_networks import language_generator, build_sampler
from data.import_data import load_captions
from data.data_processing import onehot_encode

class SplitAE(object):

    def __init__(self, dim_x, dim_z, dim_labels, dim_w, word_to_idx, lambda_x=1.0, lambda_w=1.0, lambda_y=1.0,
                     n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512, n_layers_classifier=1,):
        """ SplitAutoencoder for image features, text descriptions, and labels.

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
        self.T = dim_captions # Number of time steps in LSTM
        self.M = 256 # word embedding size
        self.H = dim_z # LSTM hidden state size
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1], name='captions')
        self.labels = tf.placeholder(tf.float32, [None, self.dim_labels], name='labels')

        # Other regularizers
        self.dropout_rate = tf.placeholder_with_default(1.0, shape=(), name='dropout_rate')

        self.build_graph()
        print("SplitAE_xwy is initialized.")

    def build_graph(self):
        """ Build computational graph.
        """

        ### Encoder
        self.z_emb = self.get_encoder(self.x)
        self.latent_rep = self.z_emb # Latent representation
        latent_rep = tf.identity(self.latent_rep, name='latent_rep') # for fetching tensor when restoring model!

        ### Feature decoder 
        self.x_recon = self.get_decoder(self.z_emb)
        image_feature_recon = tf.identity(self.x_recon, name='image_feature_recon') 
        # Sum of squares loss
        self.x_rec_loss = tf.reduce_sum((self.x - self.x_recon)**2, 1) * self.lambda_x

        ### Text description decoder 
        self.language_loss, self.all_h_train = self.get_text_loss(self.z_emb)
        # Categorical cross-entropy loss
        self.caption_rec_loss = tf.reduce_sum(self.language_loss, 1) * self.lambda_w

        # Classifier
        self.logits = self.get_classifier(self.z_emb)
        self.clf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits) * self.lambda_y

        self.correct_pred = tf.equal(tf.argmax(self.labels, axis=-1), tf.argmax(self.logits, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        ### Total loss   
        self.loss = -tf.reduce_mean(self.x_rec_loss + self.caption_rec_loss + self.clf_loss)

        ### Evaluation ###
        self.z_emb_eval = self.get_encoder( self.x, reuse = True )

        self.x_recon_eval = self.get_decoder( self.z_emb_eval, reuse = True )
        self.sampled_captions, all_hidden = self.get_text_sampler(self.z_emb_eval, reuse=True)
        caption_sampler = tf.identity(self.sampled_captions, name='text_sampler')
        
        mask = tf.cast(tf.not_equal(tf.reduce_sum(all_hidden, axis=-1), 0), tf.float32)
        den = tf.expand_dims(tf.reduce_sum(mask, axis=-1), axis=1)
        self.text_rep = tf.reduce_sum(all_hidden, axis=1) / den 

        ### Used in training feed_dict
        self.log_var = [self.loss, self.x_rec_loss, self.caption_rec_loss, self.clf_loss, self.accuracy]
        self.val_log_var = self.log_var

    def get_encoder(self, x, y=None, reuse=None):
        """ Build encoder for image features.
        """
        with tf.variable_scope('encoder', reuse=reuse):

            if y is not None:
                x = tf.concat([x, y], 1)
            h = x
            for i in range(self.n_layers_encoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                h = tf.nn.leaky_relu(h)
            z = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_mu')
        return z 

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
        """ Resuing decoder for sampling text descriptions.
        """
        dim_h = self.H
        word_to_idx = self.word_to_idx
        T = self.T
        dim_emb = self.M

        with tf.variable_scope('model', reuse=reuse):
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
            x_loss += np.sum(train_step_results[2])
            w_loss += np.sum(train_step_results[3])
            clf_loss += np.sum(train_step_results[4])
            accuracy += np.sum(train_step_results[5])

        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            x_loss += np.sum(val_step_results[1])
            w_loss += np.sum(val_step_results[2])
            clf_loss += np.sum(val_step_results[3])
            accuracy += np.sum(val_step_results[4])

        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples, 'w_loss': w_loss / n_examples,
                'clf_loss': clf_loss / n_examples, 'accuracy': accuracy / n_batches}
    return results