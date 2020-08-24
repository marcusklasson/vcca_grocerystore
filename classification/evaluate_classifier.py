
import numpy as np
import tensorflow as tf

from classification.softmax_classifier import Classifier
from data.data_processing import onehot_encode
from utils.metrics import compute_accuracy

def evaluate_class_label_decoder(args, session, tensors, data):
    """ Evaluate class label decoder from loaded model.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        session: Tensorflow session. 
        tensors: Tensors for prediction with class label decoder.
        data: The dataset for evaluation.

    Returns:
        The fine-grained accuracy, coarse-grained accuracy, and 
        the predicted labels for each sample. 

    """
    # Get tensors
    x_tensor = tensors['x']
    labels_tensor = tensors['labels']
    scores_tensor = tensors['scores']
    accuracy_tensor = tensors['accuracy']
    posterior_samples_tensor = tensors['posterior_samples']

    # Get data
    features = data['features']
    labels = data['labels']
    n_classes = data['n_classes']
    finegrained_to_coarse = data['finegrained_to_coarse']
    n_coarse_classes = data['n_coarse_classes']
    coarse_labels = data['coarse_labels']

    batch_size = args.batch_size
    K = args.K

    n_examples = len(features)
    n_batches = int(np.ceil(n_examples/batch_size))
    predicted_labels = np.zeros(n_examples)
    accuracy = 0.

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if end > n_examples:
            end = n_examples

        x_batch = features[start:end]
        labels_batch = onehot_encode(labels[start:end], n_classes)
        acc, predicted = session.run([accuracy_tensor, scores_tensor],
                                        feed_dict={x_tensor: x_batch, labels_tensor: labels_batch, posterior_samples_tensor: K} )
        accuracy += np.sum(acc)
        predicted_labels[start:end] = np.argmax(predicted, axis=1)
    accuracy = accuracy / n_batches

    predicted_coarse_labels = np.array([finegrained_to_coarse[c] for c in predicted_labels])
    accuracy_coarse = compute_accuracy(coarse_labels, predicted_coarse_labels)
    
    print("Accuracy: {:.3f} Coarse Accuracy: {:.3f}".format(accuracy, accuracy_coarse))
    return accuracy, accuracy_coarse, predicted_labels

def evaluate_softmax_classifier(args, eval_data, train_data, train=True):
    """ Evaluate trained softmax classifier.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        eval_data: The dataset for evaluation.
        train_data: The dataset for training.

    Returns:
        The fine-grained accuracy, coarse-grained accuracy, and 
        the predicted labels for each sample. 

    """
    latent_features_train = train_data['latent_features']
    labels_train = train_data['labels']
    latent_features_eval = eval_data['latent_features']
    labels_eval = eval_data['labels']
    coarse_labels_eval = eval_data['coarse_labels']
    
    n_classes = eval_data['n_classes']
    finegrained_to_coarse = eval_data['finegrained_to_coarse']
    n_coarse_classes = eval_data['n_coarse_classes']

    inputdim = latent_features_train.shape[-1]

    clf = Classifier(latent_features_train, labels_train, n_classes, inputdim,
                     logdir=args.log_dir, modeldir=args.clf_dir, _nepoch=100)
    if train:
        clf.train()
    accuracy, predicted_labels = clf.val(latent_features_eval, labels_eval)

    predicted_coarse_labels = np.array([finegrained_to_coarse[c] for c in predicted_labels])
    accuracy_coarse = compute_accuracy(coarse_labels_eval, predicted_coarse_labels)

    print("Accuracy: {:.3f} Coarse Accuracy: {:.3f}".format(accuracy, accuracy_coarse))
    return accuracy, accuracy_coarse, predicted_labels