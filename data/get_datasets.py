
import numpy as np
from data.data_processing import load_pickle, standardization

def get_datasets(args):
    """Get the Grocery Store dataset splits with features.

    Args:
        args: Arguments from parser in train_grocerystore.py.

    Returns:
        Training, validation, and test set data in python dictionaries.

    """

    data_path = args.data_path
    feature_extractor_name = args.feature_extractor_name

    word_to_idx = load_pickle(data_path + '/word_to_idx.pkl') 
    captions = load_pickle(data_path + '/captions.pkl') 
    class_to_coarse = load_pickle(data_path + '/class_to_coarse.pkl') 

    def get_dataset(split):

        dataset = load_pickle(data_path + '/%s.dataset.pkl' %split)
        natural_img_paths = dataset['natural_image_path']
        class_img_paths = dataset['class_image_path']
        class_ids = dataset['class_id']
        coarse_class_ids = dataset['coarse_class_id']
        n_examples = len(dataset)

        labels = np.array(class_ids)
        coarse_labels = np.array(coarse_class_ids)
        n_classes = len(np.unique(labels))
        n_coarse_classes = len(np.unique(coarse_labels))

        features = load_pickle(data_path + '/%s.features.%s.pkl' % (split, feature_extractor_name)) 

        dataset = {'features': features, 'labels': labels, 'iconic_image_paths': class_img_paths,
            'captions': captions, 'word_to_idx': word_to_idx, 'n_classes': n_classes,
            'coarse_labels': coarse_labels, 'n_coarse_classes': n_coarse_classes,
            'finegrained_to_coarse': class_to_coarse, 'natural_image_paths': natural_img_paths}
        return dataset

    # Get datasets
    train_data = get_dataset('train')
    val_data = get_dataset('val')
    test_data = get_dataset('test')

    # Standardize features
    train_data['features'], mu, sigma = standardization(train_data['features']) 
    val_data['features'] = standardization(val_data['features'], mu, sigma) 
    test_data['features'] = standardization(test_data['features'], mu, sigma) 

    # Make a few corrections for the val set of grocerystore
    # since val set doesn't include images from all classes in train set
    val_data['n_classes'] = train_data['n_classes']
    val_data['n_coarse_classes'] = train_data['n_coarse_classes']

    return train_data, val_data, test_data