
import tensorflow as tf
import numpy as np

from data.data_processing import onehot_encode

class VCCA(object):

    def __init__(self, dim_x, dim_z, dim_labels, lambda_x=1.0, lambda_y=1.0, 
                    n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512, 
                    n_layers_classifier=1, use_batchnorm=False):
        """ VCCA for image features and labels.

        Args:
            dim_x: Image feature dimension.
            dim_z: Latent representation dimension.
            dim_labels: Label dimension (number of classes).
            lambda_x: Scaling weights for image feature view. 
            lambda_y: Scaling weights for class label view. 
            n_layers_encoder: Number of hidden layers in encoder. 
            n_layers_decoder: Number of hidden layers in decoder.
            fc_hidden_size: Dimension in hidden fully-connected layers 
            n_layers_classifier: Number of hidden layers in class label decoder.
            use_batchnorm: Use batch normalizatin in network or not.

        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_labels = dim_labels
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y # likelihood weighting terms
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.fc_hidden_size = fc_hidden_size
        self.n_layers_classifier = n_layers_classifier
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')
        self.labels = tf.placeholder(tf.float32, [None, self.dim_labels], name='labels')

        self.use_batchnorm = use_batchnorm
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        # Other parameters
        self.kl_weight = tf.placeholder_with_default(1.0, shape=(), name='kl_weight') 
        self.K = tf.placeholder_with_default(1, shape=(), name='posterior_samples') 

        self.build_graph()
        print("VCCA_xy is initialized.")

    def build_graph(self):
        """ Build computational graph.
        """

        ### Encoder
        self.z_sample, self.z_mu, self.z_log_sigma_sq = self.get_encoder(self.x)
        self.latent_rep = self.z_mu # Latent representation
        latent_rep = tf.identity(self.latent_rep, name='latent_rep') # for fetching tensor when restoring model!

        ### Feature decoder 
        self.x_recon = self.get_decoder(self.z_sample)
        image_feature_recon = tf.identity(self.x_recon, name='image_feature_recon') 
        # Sum of squares loss
        self.x_rec_loss = tf.reduce_sum((self.x - self.x_recon)**2, 1) * self.lambda_x

        # Classifier
        self.logits = self.get_classifier(self.z_sample)
        self.clf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits) * self.lambda_y

        self.correct_pred = tf.equal(tf.argmax(self.labels, axis=-1), tf.argmax(self.logits, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        ### KL divergence and ELBO
        self.kl_div = self.kl_divergence(self.z_mu, self.z_log_sigma_sq) * self.kl_weight   
        self.loss = -tf.reduce_mean(self.kl_div + self.x_rec_loss + self.clf_loss)

        ### Used in training feed_dict
        self.log_var = [self.loss, self.kl_div, self.x_rec_loss, self.clf_loss, self.accuracy]

        ### Evaluation ###
        self.z_sample_eval, _, _ = self.get_encoder( self.x, reuse = True )
        self.x_recon_eval = self.get_decoder( self.z_sample_eval, reuse = True )

        self.val_logits = self.get_classifier(self.z_sample_eval, reuse=True)
        self.val_score = tf.cond(self.K > 1, lambda: self.compute_prediction_score(self.val_logits, mode='avg'),
                                             lambda: tf.nn.softmax(self.val_logits))
        clf_val_score = tf.identity(self.val_score, name='classifier_val_softmax_score')

        self.val_correct_pred = tf.equal(tf.argmax(self.labels, axis=-1), tf.argmax(self.val_score, axis=-1))
        self.val_accuracy = tf.reduce_mean(tf.cast(self.val_correct_pred, tf.float32))
        clf_val_accuracy = tf.identity(self.val_accuracy, name='classifier_val_accuracy')

        self.val_log_var = [self.loss, self.kl_div, self.x_rec_loss, self.clf_loss, self.val_accuracy]

    def get_encoder(self, x, y=None, reuse=None):
        """ Build encoder for image features.
        """
        with tf.variable_scope('variational', reuse=reuse):

            if y is not None:
                x = tf.concat([x, y], 1)
            h = x
            for i in range(self.n_layers_encoder):
                h = tf.layers.dense(h, units=self.fc_hidden_size, activation=None, name='h' + str(i+1))
                if self.use_batchnorm:
                    h = tf.layers.batch_normalization(h, training=self.is_training, name='bn_' + str(i+1))
                h = tf.nn.leaky_relu(h)

            q_mu = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_mu')
            q_log_sigma_sq = tf.layers.dense(h, units=self.dim_z, activation=None, name='q_log_sigma_sq')
            z, q_mu, q_log_sigma_sq = self.reparameterization_trick(q_mu, q_log_sigma_sq)
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
                if self.use_batchnorm:
                    h = tf.layers.batch_normalization(h, training=self.is_training, name='bn_' + str(i+1))
                h = tf.nn.leaky_relu(h)
            x_recon = tf.layers.dense(h, units=self.dim_x, activation=None, name='x_recon')
        return x_recon

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
            #print('y_score.shape: ', score.shape)
        if mode == 'max_vote':
            vote = tf.equal(tf.reduce_max(score, axis=1, keepdims=True), score) 
            vote = tf.cast(vote, tf.float32)
            print('y_vote.shape: ', vote.shape)
            score = tf.reduce_sum(vote, 2)
        return score

    def reparameterization_trick(self, mu, log_sigma_sq):
        """ Reparameterization trick

            z = mu + sigma * epsilon
        """
        K = self.K
        mu = tf.tile(mu, [K, 1])
        log_sigma_sq = tf.tile(log_sigma_sq, [K, 1])
        sample = mu + tf.exp(0.5*log_sigma_sq) * tf.random_normal((tf.shape(mu)), 0., 1. )
        return sample, mu, log_sigma_sq

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
    n_classes = data['n_classes']
    n_examples = len(features)
    n_batches = int(np.ceil(n_examples/batch_size))
    if shuffle:
        perm = np.random.permutation(n_examples)
        features = features[perm]
        labels = labels[perm]
        
    total_loss = 0.
    x_loss = 0.
    kl_loss = 0.
    clf_loss = 0.
    accuracy = 0.
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        # Prepare batch with hyperparameters and set feed_dict
        x_batch = features[start:end]
        feed_dict={model.x: x_batch, model.kl_weight: kl_weight, model.K: 1}
        labels_batch = onehot_encode(labels[start:end], n_classes)
        feed_dict[model.labels] = labels_batch

        if mode == 'train':
            # Training step
            train_step_results = session.run([train_op] + model.log_var, feed_dict=feed_dict) 
            total_loss += train_step_results[1]
            kl_loss += np.sum(train_step_results[2])
            x_loss += np.sum(train_step_results[3])
            clf_loss += np.sum(train_step_results[4])
            accuracy += np.sum(train_step_results[5])
        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            kl_loss += np.sum(val_step_results[1])
            x_loss += np.sum(val_step_results[2])
            clf_loss += np.sum(val_step_results[3])
            accuracy += np.sum(val_step_results[4])
        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples, 'kl_loss': kl_loss / n_examples,
                'clf_loss': clf_loss / n_examples, 'accuracy': accuracy / n_batches}
    return results