
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter
from random import sample 

from data_processing import save_pickle, load_pickle
from import_data import load_natural_images

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/marcus/workspace/datasets/GroceryStoreDataset/dataset',
                    help='Directory for image dataset')
parser.add_argument('--save_dir', type=str, default='./data/processed', help='Directory for saving processed data')
parser.add_argument('--caption_length', type=int, default=24, help='Maximum number of words in text descriptions')

def process_caption_data(class_file, data_path, max_length, stop_words=None):
    """ Process captions and get iconic image paths.

    Args:
        class_file: CSV file called classes.csv from Grocery Store dataset.
        data_path: Path to Grocery Store dataset.
        max_length: Maximum number of words in captions.
        stop_words: Words to remove from captions.

    Returns:
        caption_data: Dictionary with class names, captions, and iconic image paths.
        class_to_id: Dictionary mapping class names to class index.

    """

    class_names = np.genfromtxt(class_file, dtype=str, delimiter=',', skip_header=1, usecols=0).tolist()
    class_ids = np.genfromtxt(class_file, dtype=int, delimiter=',', skip_header=1, usecols=1).tolist()
    class_image_paths = np.genfromtxt(class_file, dtype=str, delimiter=',', skip_header=1, usecols=4).tolist()
    class_captions_paths = np.genfromtxt(class_file, dtype=str, delimiter=',', skip_header=1, usecols=5).tolist()

    captions= []
    for cap_path in class_captions_paths:
        cap_path = data_path + cap_path 
        with open(cap_path, 'r') as file:
            cap = file.read().replace('\n', '')
            captions.append(cap)

    image_paths = []
    for img_path in class_image_paths:
        img_path = data_path + img_path 
        image_paths.append(img_path)

    class_to_id = dict(zip(class_names, class_ids))

    caption_data = pd.DataFrame.from_dict({'class_name': class_names, 'class_id': class_ids, 
                                            'image_path': image_paths, 'caption': captions})
    caption_data.sort_values(by='class_id', inplace=True)

    # Parse captions, remove signs and set to lower case 
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = caption.replace('etc','').replace('/',' ').replace('!','') # remove etc and / 
        caption = " ".join(caption.split())  # replace multiple spaces

        if stop_words is not None:
            caption = " ".join([word for word in caption.lower().split() if word not in stop_words])
        caption_data.at[i, 'caption'] = caption.lower()

    return caption_data, class_to_id

def build_vocab(annotations, threshold=1):
    """ Create vocabulary for text captions.

    Args:
        annotations: Dictionary with data, including text captions.
        threshold: Threshold on word occurrences in all captions.

    Returns:
        word_to_idx: Vocabulary as python dictionary.

    """
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<PAD>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3

    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: %d" %max_len)

    return word_to_idx

def build_caption_vector(annotations, word_to_idx, max_length=15):
    """ Use vocabulary to map words in caption to vocabulary indices.
    """
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contains only lower-case words

        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        
        # Cut captions if longer than max_length + start tag
        if len(cap_vec) > max_length+1:
            cap_vec = cap_vec[:max_length+1]
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<PAD>'])
        captions[i, :] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions

def build_class_vocab(class_file):
    """ Create dictionary mapping fine-grained class index to coarse-grained class index.
    """
    class_ids = np.genfromtxt(class_file, dtype=int, delimiter=',', skip_header=1, usecols=1).tolist()
    coarse_class_ids = np.genfromtxt(class_file, dtype=int, delimiter=',', skip_header=1, usecols=3).tolist()
    class_to_coarse = dict(zip(class_ids, coarse_class_ids))
    return class_to_coarse


def build_whole_dataset(annotations, image_file, data_path):
    """ Create dataset for Grocery Store dataset.

    Args:
        annotations: Dictionary with data, including text captions.
        image_file: Image paths with fine- and corase-grained class index for natural images.
        data_path: Path to Grocery Store dataset.

    Returns:
        Dataset with information about samples for data split.

    """
    img_paths = np.genfromtxt(image_file, dtype=str, delimiter=',', usecols=0).tolist()
    class_ids = np.genfromtxt(image_file, dtype=int, delimiter=',', usecols=1).tolist()
    coarse_class_ids = np.genfromtxt(image_file, dtype=int, delimiter=',', usecols=2).tolist()

    # Extend natural image paths with dataset directory path
    natural_img_paths = []
    for img_path in img_paths:
        natural_img_paths.append(os.path.join(data_path, img_path)) 

    class_names = annotations['class_name']
    class_img_paths = annotations['image_path']
    captions = annotations['caption']

    # Create dataset
    all_class_names = []
    all_class_img_paths = []
    all_captions = []
    for class_id in class_ids:
        all_class_names.append(class_names[class_id])
        all_class_img_paths.append(class_img_paths[class_id])
        all_captions.append(captions[class_id])

    # Create dictionary with all data
    dataset = pd.DataFrame.from_dict({'class_name': all_class_names, 'class_id': class_ids,
                                        'coarse_class_id': coarse_class_ids, 'natural_image_path': natural_img_paths,
                                        'class_image_path': all_class_img_paths, 'caption': all_captions})
    return dataset

def main():

    args = parser.parse_args() 

    # Local path to Grocery Store dataset
    data_path = args.data_dir

    # CSV file with class names and paths to iconic images and product descriptions
    class_file = os.path.join(data_path, 'classes.csv') 

    # Maximum length of captions
    max_length = args.caption_length
    # Threshold on words in vocabulary based on number of occurrences in prodct descriptions 
    word_count_threshold = 1

    # Process product descriptions/captions
    caption_data, class_to_id = process_caption_data(class_file, data_path, max_length, stop_words=None)

    # Build vocabulary from captions in caption_data
    word_to_idx = build_vocab(annotations=caption_data, threshold=word_count_threshold)

    # Note! If word_count_threshold > 1, then the caption in caption_data
    # will not be the same as the encoded caption vectors in captions below.  

    # Build caption vectors with ints representing a word in word_to_idx
    captions = build_caption_vector(annotations=caption_data, word_to_idx=word_to_idx, max_length=max_length)

    # Build class vocabulary, map from finegrained to coarse class
    class_to_coarse = build_class_vocab(class_file)
    
    # Use pickle to save data
    save_data_dir = args.save_dir
    
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
        print("Directory " , save_data_dir,  " created ") 

    save_pickle(caption_data, save_data_dir + '/annotations.pkl')
    save_pickle(word_to_idx, save_data_dir + '/word_to_idx.pkl')
    save_pickle(captions, save_data_dir + '/captions.pkl')
    save_pickle(class_to_coarse, save_data_dir + '/class_to_coarse.pkl')

    # Create whole dataset
    for split in ['train', 'val', 'test']:
        image_file = data_path + '/%s.txt' % (split)
        dataset = build_whole_dataset(caption_data, image_file, data_path)
        save_pickle(dataset, save_data_dir + '/%s.dataset.pkl' % (split))   
    
    

if __name__ == "__main__":
    main()
