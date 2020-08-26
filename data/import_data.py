
import numpy as np
import scipy

def load_natural_images(img_paths, img_size=[224, 224, 3]):
    """ Load batch of natural images.
    """
    img_height, img_width, n_channels = img_size
    n_imgs = len(img_paths)
    imgs = np.ndarray([n_imgs, img_height, img_width, n_channels], dtype=np.float32)

    for i, img_path in enumerate(img_paths):
        img = scipy.misc.imread(img_path, mode='RGB')
        img = scipy.misc.imresize(img, (img_height, img_width))
        imgs[i] = img / 255.0 ### This should be commented if we use VGG preprocessing!
    return imgs

def load_iconic_images(img_paths, img_size=[64, 64, 3]):
    """ Load batch of iconic images.
    """
    img_height, img_width, n_channels = img_size
    n_imgs = len(img_paths)
    imgs = np.ndarray([n_imgs, img_height, img_width, n_channels], dtype=np.float32)

    for i, img_path in enumerate(img_paths):
        img = scipy.ndimage.imread(img_path)
        img = scipy.misc.imresize(img, (img_height, img_width))
        imgs[i] = img / 255.0
    return imgs

def load_captions(captions, class_ids):
    """ Load batch of text descriptions.
    """
    n_caps = len(class_ids)
    n_seq_len = captions.shape[1] 
    out_captions = np.ndarray([n_caps, n_seq_len], dtype=np.int32)

    for i, class_id in enumerate(class_ids):
        class_id = int(class_id)
        out_captions[i,:] = captions[class_id,:]
    return out_captions