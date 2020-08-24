
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
parser.add_argument('--data_dir', type=str, default='/home/marcus/workspace/datasets/GroceryStoreDataset/dataset', help='Directory for image dataset')
parser.add_argument('--save_dir', type=str, default='/home/marcus/workspace/papers/vcca_grocerystore/data/processed', help='Directory for saving processed data')
parser.add_argument('--caption_length', type=int, default=24, help='Maximum number of words in text descriptions')

def _process_caption_data(class_file, data_path, max_length, stop_words=None):

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

        #print(caption)
        if stop_words is not None:
            caption = " ".join([word for word in caption.lower().split() if word not in stop_words])
        #print(caption)
        caption_data.set_value(i, 'caption', caption.lower())

    return caption_data, class_to_id

def _build_vocab(annotations, threshold=1, fixed_vocab_size=None):
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

    #word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    word_to_idx = {u'<PAD>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    #word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    #idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: %d" %max_len)

    if fixed_vocab_size:
        pre_vocab_size = len(word_to_idx)
        word_to_idx = {k: v for k,v in word_to_idx.items() if v < fixed_vocab_size}
        print('Reduced vocabulary size from %d to %d' %(pre_vocab_size, len(word_to_idx)))

    return word_to_idx

def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contains only lower-case words

        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        #for word in words:
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            #else: ### If word is not in vocab, append token for <UNK>
            #    cap_vec.append(word_to_idx['<UNK>'])
        
        # Cut captions if longer than max_length + start tag
        if len(cap_vec) > max_length+1:
            cap_vec = cap_vec[:max_length+1]
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                #cap_vec.append(word_to_idx['<NULL>'])
                cap_vec.append(word_to_idx['<PAD>'])
        #embed() 
        captions[i, :] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions

def _build_class_vocab(class_file):
    class_ids = np.genfromtxt(class_file, dtype=int, delimiter=',', skip_header=1, usecols=1).tolist()
    coarse_class_ids = np.genfromtxt(class_file, dtype=int, delimiter=',', skip_header=1, usecols=3).tolist()
    class_to_coarse = dict(zip(class_ids, coarse_class_ids))
    return class_to_coarse


def _build_whole_dataset(annotations, image_file, data_path):
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
    #dataset.sort_values(by='class_id', inplace=True)

    return dataset

def _validation_split(dataset, n_examples_per_class=3):

    if n_examples_per_class < 1:
        return dataset
    
    n = len(dataset)
    n_val = int(n*0.1) 

    class_ids = pd.unique(dataset['class_id']).tolist()
    train_dataset = dataset
    df = pd.DataFrame() 

    indices = []
    for i in class_ids:
        # Get indices in dataset that should be moved to validation set
        idx = train_dataset.index[train_dataset['class_id'] == i].tolist()
        idx_val = sample(idx, n_examples_per_class)
        indices = indices + idx_val

    # Slower than previously, but doesn't mess up the loading of data
    # Might have been  
    train_df = pd.DataFrame()
    val_df = pd.DataFrame() 
    for v in range(len(dataset)):
        row_df = dataset.iloc[v]
        if (v in indices):
            val_df = val_df.append(row_df, ignore_index=True) 
        else:
            #row_df = dataset.iloc[v] 
            train_df = train_df.append(row_df, ignore_index=True) 

    # Change class_id and coarse_class_id to int32
    train_df = train_df.astype({'class_id': 'int32'}) 
    train_df = train_df.astype({'coarse_class_id': 'int32'}) 

    val_df = val_df.astype({'class_id': 'int32'}) 
    val_df = val_df.astype({'coarse_class_id': 'int32'}) 

    print("Validation set size: ", len(val_df))
    print("Training set size: ", len(train_df))

    #return df, train_dataset
    return val_df, train_df


def main():

    args = parser.parse_args() 

    # Local path to Grocery Store dataset
    data_path = args.data_dir #'/home/marcus/temp_GroceryStoreDataset/dataset'
    #data_path = '/home/marcus/Workspace/GroceryStoreDataset/dataset'

    # CSV file with class names and paths to iconic images and product descriptions
    class_file = os.path.join(data_path, 'classes.csv') 

    # Maximum length of captions
    max_length = args.caption_length
    # Threshold on words in vocabulary based on number of occurrences in prodct descriptions 
    word_count_threshold = 1

    # Process product descriptions/captions
    caption_data, class_to_id = _process_caption_data(class_file, data_path, max_length, stop_words=None)

    # Build vocabulary from captions in caption_data
    word_to_idx = _build_vocab(annotations=caption_data, threshold=word_count_threshold)

    # Note! If word_count_threshold > 1, then the caption in caption_data
    # will not be the same as the encoded caption vectors in captions below.  

    # Build caption vectors with ints representing a word in word_to_idx
    captions = _build_caption_vector(annotations=caption_data, word_to_idx=word_to_idx, max_length=max_length)

    # Build class vocabulary, map from finegrained to coarse class
    class_to_coarse = _build_class_vocab(class_file)
    
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
        dataset = _build_whole_dataset(caption_data, image_file, data_path)
        save_pickle(dataset, save_data_dir + '/%s.dataset.pkl' % (split))   
    
    

if __name__ == "__main__":
    main()
