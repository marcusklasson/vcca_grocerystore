
import pickle
import numpy as np

def decode_captions(captions, idx_to_word):
    """ Decode text captions from index in vocabulary to words.
    """
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def standardization(x, mu=None, sigma=None):
    """ Feature scaling with standardization.
    """
    if mu is None or sigma is None:
        if len(x.shape) > 2:
            mu = np.mean(x, axis=(0, 1, 2))
            sigma = np.std(x, axis=(0, 1, 2))
            #eps = 1e-8 # For numerical stability
        else:
            eps = 1e-8 # For numerical stability
            mu = np.mean(x, axis=0)
            sigma = np.std(x, axis=0) + eps
        return (x - mu)/sigma, mu, sigma
    return (x - mu)/sigma

def onehot_encode(y, n_classes):
    """
    Onehot encoding of scalars in a list or 1D numpy array 
    """
    n = len(y)
    if len(y.shape) == 2:
        y = np.squeeze(y, axis=-1)
    y_onehot = np.zeros([n, n_classes], dtype=float)
    y_onehot[np.arange(n), y] = 1.0
    return y_onehot
