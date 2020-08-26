
import tensorflow as tf
import numpy as np
import sys
import os, os.path
import random
import argparse
import pickle
import datetime 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def classificationLayer(x, classes, name="classification", reuse=False, isTrainable=True, seed=0):
    """ Create linear layer as classifier.
    """
       
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(x, units=classes, name='clf_output', activation=None,
                            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02, seed=seed),
                            bias_initializer=tf.zeros_initializer(),
                            trainable=isTrainable, reuse=reuse)
        net = tf.reshape(net, [-1, classes])    
    return net

class Classifier:
    # train_Y is integer 
    def __init__(self, _train_X, _train_Y,  _nclass, _input_dim, logdir, modeldir,
                 _lr=0.001, _beta1=0.5, _nepoch=100, _batch_size=100, pretrain_classifer='',
                 seed=0, save_model=True, _val_X=None, _val_Y=None):
        """ Initialize softmax classifier.

            Code adapted from:
            https://github.com/akku1506/Feature-Generating-Networks-for-ZSL/blob/master/classifier.py

        Args:
            _train_X: Features from training set. 
            _train_Y: Labels for training set.
            _nclass: Number of classes in dataset for setting output of classifier.
            _input_dim: Input data dimension.
            logdir: Directory where to save logs.
            modeldir: Directory for saving classifier.
            _lr: Learning rate.
            _beta1: Hyperparameter for Adam optimizer.
            _nepoch: Number of training epochs.
            _batch_size: Batch size.
            pretrain_classifier: 
            seed: Random seed.
            save_model: Option for saving classifier or not.
            _val_X: Features from validation set.
            _val_Y: Labels from validation set.

        """
        self.train_X = _train_X 
        self.train_Y = _train_Y 
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _input_dim
        self.lr = _lr
        self.beta1 = _beta1
        
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.shape[0]
        self.logdir = os.path.join(logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
        self.modeldir = modeldir

        # New parameters
        self.seed = seed
        self.save_model = save_model
        self.val_X =  _val_X 
        self.val_Y = _val_Y

        # Set seed
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        ##########model_definition
        self.input = tf.placeholder(tf.float32, [None, self.input_dim], name='input')
        self.label =  tf.placeholder(tf.int32, [None], name='label')
        if pretrain_classifer == '':        
            #self.classificationLogits = classificationLayer(self.input, self.nclass)
            self.classificationLogits = classificationLayer(self.input, self.nclass, seed=self.seed)
            ############classification loss#########################

            self.classificationLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classificationLogits,
                                                                                                    labels=self.label))
            classifierParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classification')
            
            self.classificationAcc = tf.reduce_mean(tf.cast(
                                        tf.equal(tf.argmax(self.classificationLogits, axis=1, output_type=tf.int32), self.label),
                                        tf.float32))
            print ('...................')

            classifierOptimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, beta2=0.999)
      
            classifierGradsVars = classifierOptimizer.compute_gradients(self.classificationLoss, var_list=classifierParams)    
            self.classifierTrain = classifierOptimizer.apply_gradients(classifierGradsVars)

            #################### what all to visualize  ############################
            tf.summary.scalar("ClassificationLoss", self.classificationLoss)
            tf.summary.scalar("ClassificationAcc", self.classificationAcc)
            for g,v in classifierGradsVars:    
                tf.summary.histogram(v.name,v)
                tf.summary.histogram(v.name+str('grad'),g)

            # Initialize savers!
            self.saver = tf.train.Saver(max_to_keep=1)
            self.merged_all = tf.summary.merge_all()        
    
    def next_batch(self, batch_size):
        """ Fetching next batch in dataset.
        """
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = np.random.permutation(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            
            self.index_in_epoch = 0
            self.epochs_completed += 1
            return X_rest_part, Y_rest_part
            
            # shuffle the data
            perm = np.random.permutation(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return np.concatenate((X_rest_part, X_new_part), axis=0), np.concatenate((Y_rest_part, Y_new_part), axis=0)
            else:
                return X_new_part, Y_new_part
            
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def train(self):
        """ Train classifier and optionally evaluate on validation set.
        """
        print('Start training classifier...')
        # Set seed for dataset shuffle
        np.random.seed(self.seed)    

        k=1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Initialize summary writer!
            summary_writer = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
            val_writer = tf.summary.FileWriter(self.logdir + '/val')
            for epoch in range(1, self.nepoch+1):
                total_loss = 0.0
                total_acc = 0.0
                n_batches = 0.0
                for i in range(0, self.ntrain, self.batch_size):      
                    batch_input, batch_label = self.next_batch(self.batch_size)
                    _, loss, acc, merged, logits = sess.run([self.classifierTrain, self.classificationLoss, 
                                                        self.classificationAcc, self.merged_all,
                                                        self.classificationLogits], 
                                                        feed_dict={self.input: batch_input, self.label: batch_label}) 
                    total_loss += loss
                    total_acc += acc
                    n_batches += 1
                    predicted_label = np.argmax(logits, axis=1)
                    # Add results to summary!
                    if i == 0:
                        summary_writer.add_summary(merged, epoch)
                        k=k+1
                if (epoch % 10) == 0:
                    print('Epoch {:d}, Loss: {:.4f} Accuracy: {:.4f} '.format(epoch, 
                                                                            total_loss / n_batches,
                                                                            total_acc / n_batches))
                # Validation
                if self.val_X is not None and self.val_Y is not None:
                    nval = self.val_X.shape[0]
                    start = 0
                    total_loss = 0.0
                    total_acc = 0.0
                    n_batches = 0.0
                    for i in range(0, nval, self.batch_size): 
                        end = min(nval, start + self.batch_size)
                        batch_input = self.val_X[start:end]
                        batch_label = self.val_Y[start:end]
                        if len(batch_input.shape) == 1:
                            batch_input = load_features(batch_input, self.input_dim) 

                        loss, acc, merged = sess.run([self.classificationLoss, self.classificationAcc, self.merged_all], 
                                                        feed_dict={self.input: batch_input, self.label: batch_label}) 
                        total_loss += loss
                        total_acc += acc
                        n_batches += 1
                        # Add results to summary!
                        if i == 0:
                            val_writer.add_summary(merged, epoch)
                            k=k+1    
                    print('Validation, Loss: ${:.4f} Accuracy: {:.4f} \n'.format(total_loss / n_batches,
                                                                               total_acc / n_batches))                  

                # Saving classifiers
                if self.save_model:
                    self.saver.save(sess, os.path.join(self.modeldir, 'models_'+str(epoch)+'.ckpt'))        

    def val(self, test_X, test_label): 
        """ Load saved classifier and evaluate on test data.
        """
        start = 0
        ntest = test_X.shape[0]
        predicted_label = np.empty_like(test_label)
        
        self.input = tf.placeholder(tf.float32,[None, self.input_dim], name='eval_inputs')
 
        self.classificationLogits = classificationLayer(self.input, self.nclass, reuse=True, 
                                                        isTrainable=False, seed=self.seed)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classification')
            
            self.saver = tf.train.Saver(var_list=params)
            string = self.modeldir + '/models_'+str(self.nepoch)+'.ckpt'
            print ('Classifier checkpoint: ', string) 
            try:
                self.saver.restore(sess, string)
            except:
                print("Previous weights not found of classifier") 
                sys.exit(0)

            print ("Classifier loaded")
            self.saver = tf.train.Saver()

            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                batch_input = test_X[start:end]
                output = sess.run([self.classificationLogits], feed_dict={self.input: batch_input}) 
                predicted_label[start:end] = np.argmax(np.squeeze(np.array(output)), axis=1)
                start = end

            acc = np.mean((test_label == predicted_label).astype(float))
            return acc, predicted_label
