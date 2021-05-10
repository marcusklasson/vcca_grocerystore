
import os
import argparse
import numpy as np
import tensorflow as tf

from data.get_datasets import get_datasets
from classification.softmax_classifier import Classifier
from utils.metrics import compute_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/processed', help='Data directory')
parser.add_argument('--model_dir', type=str, default='./softmax_clf', help='Saved model directory')
parser.add_argument('--feature_extractor_name', type=str, default='densenet', help='Feature extractor.')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--save_file', type=str, default='./clf_metrics.txt', help='File for saving results')
args = parser.parse_args() 

### Simple script for training Softmax classifier 
### on off-the-shelf DenseNet features.
### Similar as code in evaluate_classifier.py.

# Set random seed
tf.reset_default_graph()
print("Random seed: ", args.seed)
tf.random.set_random_seed(args.seed)
np.random.seed(args.seed)

### Load datasets
train_data, _, test_data = get_datasets(args)

n_classes = train_data['n_classes']
finegrained_to_coarse = train_data['finegrained_to_coarse']
n_coarse_classes = train_data['n_coarse_classes']

# Train classifier
inputdim = train_data['features'].shape[-1]
log_dir = os.path.join(args.model_dir, 'logs')
clf = Classifier(train_data['features'], train_data['labels'], n_classes, inputdim,
                 logdir=log_dir, modeldir=args.model_dir, _nepoch=100, seed=args.seed)
clf.train()
accuracy, predicted_labels = clf.val(test_data['features'], test_data['labels'])

predicted_coarse_labels = np.array([finegrained_to_coarse[c] for c in predicted_labels])
accuracy_coarse = compute_accuracy(test_data['coarse_labels'], predicted_coarse_labels)

print("Accuracy: {:.3f} Coarse Accuracy: {:.3f}".format(accuracy, accuracy_coarse))
with open(args.save_file, 'a') as file:
    file.write("Seed  Accuracy  Coarse Accuracy  \n")
    file.write("{:d}  {:.3f}    {:.3f}  \n".format(args.seed, accuracy, accuracy_coarse))