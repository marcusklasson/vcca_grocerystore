
import tensorflow as tf
import numpy as np

from data.import_data import load_iconic_images
from data.data_processing import onehot_encode

from models.networks.conv_networks import conv_decoder

class SplitAE(object):

    def __init__(self, dim_x, dim_z, dim_i, dim_labels, 
                    lambda_x=1.0, lambda_i=1.0, lambda_y=1.0,
                    n_layers_encoder=1, n_layers_decoder=1, fc_hidden_size=512, n_layers_classifier=1,):
        """ SplitAutoencoder for image features, iconic images, and labels.

        Args:
            dim_x: Image feature dimension.
            dim_z: Latent representation dimension.
            dim_i: Iconic image dimension.
            dim_labels: Label dimension (number of classes).
            lambda_x: Scaling weights for image feature view. 
            lambda_i: Scaling weights for iconic image view. 
            lambda_y: Scaling weights for class label view. 
            n_layers_encoder: Number of hidden layers in encoder. 
            n_layers_decoder: Number of hidden layers in decoder.
            fc_hidden_size: Dimension in hidden fully-connected layers 
            n_layers_classifier: Number of hidden layers in class label decoder.

        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_i = dim_i
        self.dim_labels = dim_labels

        self.lambda_x = lambda_x
        self.lambda_i = lambda_i
        self.lambda_y = lambda_y
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.fc_hidden_size = fc_hidden_size
        self.n_layers_classifier = n_layers_classifier
        
        self.x = tf.placeholder(tf.float32, [None, self.dim_x], name='x')
        self.iconic_images = tf.placeholder(tf.float32, [None, dim_i[0], dim_i[1], dim_i[2]], name='i')
        self.labels = tf.placeholder(tf.float32, [None, self.dim_labels], name='labels')

        # Used batch normalization in generator
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        self.build_graph()
        print("SplitAE_xiy is initialized.")

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

        ### Iconic image decoder 
        self.i_recon = self.get_conv_decoder( self.z_emb )
        iconic_image_recon = tf.identity(self.i_recon, name='iconic_image_recon') 
        # Sum of squares loss
        self.i_rec_loss = tf.reduce_sum((self.iconic_images - self.i_recon)**2, [1, 2, 3]) * self.lambda_i

        # Classifier
        self.logits = self.get_classifier(self.z_emb)
        logits = tf.identity(self.logits, name='logits')
        self.clf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits) * self.lambda_y

        self.correct_pred = tf.equal(tf.argmax(self.labels, axis=-1), tf.argmax(self.logits, axis=-1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        accuracy = tf.identity(self.accuracy, name='accuracy')

        ### Total loss
        self.loss = -tf.reduce_mean(self.x_rec_loss + self.i_rec_loss + self.clf_loss)

        ### Used in training feed_dict
        self.log_var = [self.loss, self.x_rec_loss, self.i_rec_loss, self.clf_loss, self.accuracy]
        self.val_log_var = self.log_var

        ### Evaluation ###
        self.z_emb_eval = self.get_encoder(self.x, reuse=True)
        self.x_recon_eval = self.get_decoder(self.z_emb_eval, reuse=True)
        self.i_recon_eval = self.get_conv_decoder(self.z_emb_eval, reuse=True) 

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

            z = tf.layers.dense(h, units=self.dim_z, activation=None, name='z')
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

    def get_conv_decoder(self, z, reuse=None):
        """ Build decoder for iconic image.
        """
        is_training = self.is_training
        with tf.variable_scope('model', reuse=reuse):
            y_recon = conv_decoder(z, is_training=is_training, reuse=reuse)
        return y_recon

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
    clf_loss = 0.
    accuracy = 0.
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        # Prepare batch and hyperparameters 
        x_batch = features[start:end]
        i_batch = load_iconic_images(iconic_image_path[start:end])
        labels_batch = onehot_encode(labels[start:end], n_classes)
        feed_dict={model.x: x_batch, model.iconic_images: i_batch, 
                    model.labels: labels_batch, model.is_training: is_training}

        if mode == 'train':
            # Training step
            train_step_results = session.run([train_op] + model.log_var, feed_dict=feed_dict) 
            total_loss += train_step_results[1]
            x_loss += np.sum(train_step_results[2])
            i_loss += np.sum(train_step_results[3])
            clf_loss += np.sum(train_step_results[4])
            accuracy += np.sum(train_step_results[5])

        elif mode == 'val':
            # Validation step
            val_step_results = session.run(model.val_log_var, feed_dict=feed_dict)
            total_loss += val_step_results[0]
            x_loss += np.sum(val_step_results[1])
            i_loss += np.sum(val_step_results[2])
            clf_loss += np.sum(val_step_results[3])
            accuracy += np.sum(val_step_results[4])

        else:
            raise ValueError("Argument \'mode\' %s doesn't exist!" %mode)

    # Epoch finished, return results. clf_loss and accuracy are zero if args.classifier_head is False
    results = {'total_loss': total_loss / n_batches, 'x_loss': x_loss / n_examples, 'i_loss': i_loss / n_examples, 
                'clf_loss': clf_loss / n_examples, 'accuracy': accuracy / n_batches}
    return results