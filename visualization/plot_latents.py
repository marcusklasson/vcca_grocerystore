
import os
import argparse

import numpy as np
import matplotlib 
# had to change backend for matplotlib from XWindows to Agg for plotting to work on cluster
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from data.data_processing import load_pickle

def plot_latent_representation(args, train_data, test_data, method='pca'):
    """ Plot latent representations in 2D.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        train_data: Training dataset used for fitting pca space.
        test_data: Test dataset to be visualized.
        method: Visualization method (pca or tsne)

    """

    print('Visualize latent reprentations...')
    if method == 'pca':
        print("Run PCA... ")
        pca = PCA(n_components=2)
        pca.fit(train_data['latents'])
        transformed = pca.transform(test_data['latents'])

    elif method == 'tsne':
        print("Run TSNE...")
        transformed = TSNE(n_components=2).fit_transform(test_data['latents'])
    else:
        ValueError('Unknown plotting method: %s'.format(method))

    # Plot all ttypes of images
    plot_with_iconic_images(args, transformed, test_data, method=method)
    label2classname = get_label_to_class_name_dict(args.data_path)
    plot_with_apples(args, transformed, test_data['labels'], label2classname, method=method)
    plot_with_juice_yoghurt(args, transformed, test_data['labels'], label2classname, method=method)
    plot_with_peppers(args, transformed, test_data['labels'], label2classname, method=method)

    # Plot private latent spaces for VCCA_private
    if args.use_private:
        pca.fit(train_data['latents_ux'])
        transformed_ux = pca.transform(test_data['latents_ux'])
        plot_with_natural_images(args, transformed_ux, test_data, method=method, var='ux')

        if args.use_text:
            pca.fit(train_data['latents_uw'])
            transformed_uw = pca.transform(test_data['latents_uw'])
            plot_with_iconic_images(args, transformed_uw, test_data, method=method, var='uw', max_number=1)

        if args.use_iconic:
            pca.fit(train_data['latents_ui'])
            transformed_ui = pca.transform(test_data['latents_ui'])
            plot_with_iconic_images(args, transformed_ui, test_data, method=method, var='ui', max_number=1)

def plot_with_iconic_images(args, transformed, test_data, method='pca', var='z', max_number=10):
    """ Plot latent representations as their corresponding iconic images in 2D.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        transformed: 2D features transformed by visualization method.
        test_data: Test dataset with information for plotting.
        method: Visualization method (pca or tsne)
        max_number: Maximum number of latents to be plotted for each class.

    """    
    print("Plot transformed embeddings with iconic images...")
    fig, ax = plt.subplots()
    ax.scatter(transformed[:, 0], transformed[:, 1], c='white')    

    iconic_image_paths = test_data['iconic_image_paths']
    labels = test_data['labels']
    count_dict = dict(zip(np.unique(labels), 0*np.unique(labels))) 

    # Size settings of iconic image in plot
    sz=32
    zoom=0.3
    for x0, y0, class_id, img_path in zip(transformed[:, 0], transformed[:, 1], labels, iconic_image_paths):
        if count_dict[class_id] < 10:
            count_dict[class_id] += 1
            ab = AnnotationBbox(get_image(img_path, size=sz, zoom=zoom), (x0, y0), frameon=False, pad=1.0)
            ax.add_artist(ab)

    plt.axis('off')
    plt.tight_layout()
    file_name = '%s_latents_%s_%s_seed%d.png' %(method, var, args.model_name, args.seed)
    fig.savefig(os.path.join(args.save_dir, file_name))

def plot_with_natural_images(args, transformed, test_data, method='pca', var='z', max_number=10):
    """ Plot latent representations as their corresponding natural images in 2D.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        transformed: 2D features transformed by visualization method.
        test_data: Test dataset with information for plotting.
        method: Visualization method (pca or tsne)
        max_number: Maximum number of latents to be plotted for each class.

    """    
    print("Plot transformed embeddings with natural images...")
    fig, ax = plt.subplots()
    ax.scatter(transformed[:, 0], transformed[:, 1], c='white')

    natural_image_paths = test_data['natural_image_paths']
    labels = test_data['labels']
    count_dict = dict(zip(np.unique(labels), 0*np.unique(labels))) 

    sz=128
    zoom=0.1
    for x0, y0, class_id, img_path in zip(transformed[:, 0], transformed[:, 1], labels, natural_image_paths):
        if count_dict[class_id] < 10:
            count_dict[class_id] += 1
            ab = AnnotationBbox(get_image(img_path, size=sz, zoom=zoom), (x0, y0), frameon=False, pad=1.0)
            ax.add_artist(ab)

    plt.axis('off')
    plt.tight_layout()
    file_name = '%s_latents_%s_%s_seed%d.png' %(method, var, args.model_name, args.seed)
    fig.savefig(os.path.join(args.save_dir, file_name))

def plot_with_apples(args, transformed, labels, label2name, method='pca'):
    """ Plot latent representations in 2D with emphasis
        on the apple classes in the dataset.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        transformed: 2D features transformed by visualization method.
        labels: Corresponding labels to transformed features.
        label2name: Dictionary mapping label to class name.
        method: Visualization method (pca or tsne)

    """  
    print("Plot transformed embeddings of red and green apples...")

    fig, ax = plt.subplots()

    red_apples_colors = {'Pink-Lady': 'tomato', 'Red-Delicious': 'tomato', 'Royal-Gala': 'tomato'}
    green_apples_colors = {'Golden-Delicious': 'darkgreen', 'Granny-Smith': 'darkgreen'}

    name2label = {v: k for k, v in label2name.items()} 

    red_apples_ind = []
    red_apples_feats = []
    for k in red_apples_colors.keys():
        v = name2label[k]
        ind = np.where(labels==v)[0]
        red_apples_ind.append(ind)
        red_apples_feats.append(transformed[ind])

    green_apples_ind = []
    green_apples_feats = []
    for k in green_apples_colors.keys():
        v = name2label[k]
        ind = np.where(labels==v)[0]
        green_apples_ind.append(ind)
        green_apples_feats.append(transformed[ind])

    # Copy transformed and remove indices of juice and youghrt packages
    ind = np.array([], dtype=int) 
    for i in range(0, len(red_apples_ind)):
        ind = np.append(ind, red_apples_ind[i])
    for i in range(0, len(green_apples_ind)):
        ind = np.append(ind, green_apples_ind[i])
    transformed1 = transformed 
    transformed1 = np.delete(transformed1, ind, axis=0)

    ax.scatter(x=transformed1[:, 0], y=transformed1[:, 1], s=10, edgecolors='blue', facecolors='none', alpha=0.3)
    for feats in red_apples_feats:
        ax.scatter(x=feats[:, 0], y=feats[:, 1], s=80, c='tomato', alpha=0.9, )
    for feats in green_apples_feats:
        ax.scatter(x=feats[:, 0], y=feats[:, 1], s=80, c='darkgreen', alpha=0.9, )

    plt.axis('off')
    plt.tight_layout()
    file_name = '%s_latent_apples_%s_seed%d.png' %(method, args.model_name, args.seed)
    fig.savefig(os.path.join(args.save_dir, file_name))

def plot_with_juice_yoghurt(args, transformed, labels, label2name, method='pca'):
    """ Plot latent representations in 2D with emphasis
        on juice and yoghurt classes with light-colored
        package textures in the dataset.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        transformed: 2D features transformed by visualization method.
        labels: Corresponding labels to transformed features.
        label2name: Dictionary mapping label to class name.
        method: Visualization method (pca or tsne)

    """  
    print("Plot transformed embeddings of juice and yoghurt packages...")

    fig, ax = plt.subplots()

    juice_colors = {'Tropicana-Apple-Juice': 'gold', 'Tropicana-Golden-Grapefruit': 'gold',
                         'Tropicana-Juice-Smooth': 'gold', 'Tropicana-Mandarin-Morning': 'gold'}
    yoghurt_colors = {'Valio-Vanilla-Yoghurt': 'darkgreen', 'Yoggi-Strawberry-Yoghurt': 'darkgreen',
                         'Yoggi-Vanilla-Yoghurt': 'darkgreen'}
    name2label = {v: k for k, v in label2name.items()} 

    juice_ind = []
    juice_feats = []
    for k in juice_colors.keys():
        v = name2label[k]
        ind = np.where(labels==v)[0]
        juice_ind.append(ind)
        juice_feats.append(transformed[ind])

    yoghurt_ind = []
    yoghurt_feats = []
    for k in yoghurt_colors.keys():
        v = name2label[k]
        ind = np.where(labels==v)[0]
        yoghurt_ind.append(ind)
        yoghurt_feats.append(transformed[ind])

    # Copy transformed and remove indices of juice and youghrt packages
    ind = np.array([], dtype=int) 
    for i in range(0, len(juice_ind)):
        ind = np.append(ind, juice_ind[i])
    for i in range(0, len(yoghurt_ind)):
        ind = np.append(ind, yoghurt_ind[i])
    transformed1 = transformed 
    transformed1 = np.delete(transformed1, ind, axis=0)

    ax.scatter(x=transformed1[:, 0], y=transformed1[:, 1], s=10, edgecolors='blue', facecolors='none', alpha=0.3)

    for feats in juice_feats:
        ax.scatter(x=feats[:, 0], y=feats[:, 1], s=80, c='gold', alpha=0.9, )
    for feats in yoghurt_feats:
        ax.scatter(x=feats[:, 0], y=feats[:, 1], s=80, c='green', alpha=0.9, )

    plt.axis('off')
    plt.tight_layout()
    file_name = '%s_latent_juice_yoghurt_%s_seed%d.png' %(method, args.model_name, args.seed)
    fig.savefig(os.path.join(args.save_dir, file_name))

def plot_with_peppers(args, transformed, labels, label2name, method='pca'):
    """ Plot latent representations in 2D with emphasis
        on the bell pepper classes in the dataset.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        transformed: 2D features transformed by visualization method.
        labels: Corresponding labels to transformed features.
        label2name: Dictionary mapping label to class name.
        method: Visualization method (pca or tsne)

    """  
    print("Plot transformed embeddings of peppers...")

    fig, ax = plt.subplots(1, 1)

    bell_pepper_colors = {'Green-Bell-Pepper': 'darkgreen', 'Orange-Bell-Pepper': 'orange', 
                            'Red-Bell-Pepper': 'tomato', 'Yellow-Bell-Pepper': 'gold'}
    green_bp = np.where(labels==69)[0]
    orange_bp = np.where(labels==70)[0]
    red_bp = np.where(labels==71)[0]
    yellow_bp = np.where(labels==72)[0]
    ind = np.concatenate((green_bp, yellow_bp, red_bp, orange_bp))

    green_feats = transformed[green_bp]
    orange_feats = transformed[orange_bp]
    red_feats = transformed[red_bp]
    yellow_feats = transformed[yellow_bp]

    # Remove indices of bell peppers from transformed
    transformed1 = transformed
    transformed1 = np.delete(transformed, ind, axis=0)

    ax.scatter(x=transformed1[:, 0], y=transformed1[:, 1], s=10, edgecolors='blue', facecolors='none', alpha=0.3)
    ax.scatter(x=green_feats[:, 0], y=green_feats[:, 1], s=80, c='darkgreen', alpha=0.9, )
    ax.scatter(x=orange_feats[:, 0], y=orange_feats[:, 1], s=80, c='orange', alpha=0.9)
    ax.scatter(x=red_feats[:, 0], y=red_feats[:, 1], s=80, c='tomato', alpha=0.9)
    ax.scatter(x=yellow_feats[:, 0], y=yellow_feats[:, 1], s=80, c='gold', alpha=0.9)

    #plt.axis('off')
    if method == 'pca':
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
    
    plt.tight_layout()
    file_name = '%s_latent_peppers_%s_seed%d.png' %(method, args.model_name, args.seed)
    fig.savefig(os.path.join(args.save_dir, file_name))
    #plt.show()

def get_image(path, size=32, zoom=0.3):
    """ Get image for plotting on top of 2D points.
    """
    return OffsetImage(Image.open(path).resize([size,size]), zoom=zoom)

def get_label_to_iconic_image_dict(path, split='test'):
    """ Get dictionary mapping labels to iconic image paths.
    """
    dataset = load_pickle(path + '/%s.dataset.pkl' %split)
    labels = dataset['class_id'].unique().tolist()
    iconic_image_paths = dataset['class_image_path'].unique().tolist()
    return dict(zip(labels, iconic_image_paths))

def get_label_to_class_name_dict(path, split='test'):
    """ Get dictionary mapping labels to class names.
    """
    dataset = load_pickle(path + '/%s.dataset.pkl' %split)
    labels = dataset['class_id'].unique().tolist()
    class_names = dataset['class_name'].unique().tolist()
    return dict(zip(labels, class_names))    