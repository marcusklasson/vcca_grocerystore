
import tensorflow as tf

weight_initializer = tf.contrib.layers.xavier_initializer()
const_initializer = tf.constant_initializer(0.0)
emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

def language_generator(z, captions, word_to_idx, T, dim_h, dim_emb, dropout_rate=1.0, word_dropout_rate=1.0, reuse=None):
    """Get LSTM network for decoding text descriptions.

    Args:
        z: Input embedding for initializing the hidden states.
        captions: Text captions for supervision.
        word_to_idx: Vocabulary encoding words to index.
        T: Number of steps for LSTM.
        dim_h: Hidden state dimension.
        dim_emb: Word embedding dimension.
        dropout_rate: Dropout rate before linear layer.
        word_dropout_rate: Dropout rate for masking out words.
        reuse: Indicates if variable scopes should be reused or to create new scopes.

    Returns:
        Computed cross-entropy loss and all hidden states.

    """
    pad_tag = word_to_idx['<PAD>']
    vocab_size = len(word_to_idx)

    # Define in and out captions and mask
    captions_in = captions[:, :T]
    captions_out = captions[:, 1:]
    mask = tf.cast(tf.not_equal(captions_out, pad_tag), tf.float32) 
    h_mask = tf.cast(tf.not_equal(captions_in, pad_tag), tf.float32)

    # For storing all hidden states
    all_hidden = []

    # Initialize LSTM states
    c, h = _get_initial_lstm(z, dim_h)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=dim_h)

    # Word dropout as in Bowman et al. (2015)
    start_tag = word_to_idx['<START>']
    # All captions are kept the same if word_dropout_rate = 1.0
    # If word_dropout_rate = 0.0, then every word in caption will be set to <UNK> tag
    prob = tf.random.uniform(tf.shape(captions_in)) # Sample probability for setting word to <UNK>
    mask_start = tf.cast(tf.not_equal(captions_in, start_tag), tf.float32) # Mask for start tags, such that we don't set these to <UNK>
    mask_pad = tf.cast(tf.not_equal(captions_in, pad_tag), tf.float32) # Mask for padding tags, such that we don't set these to <UNK>
    prob_masked = prob * mask_start * mask_pad # Set probabilities for start and padding tags to zero, such that we don't set these to <UNK>
    # Create mask for <UNK> tags, if probability is larger than word_dropout_rate then set these words to <UNK>
    # For this code to be valid, then the vocabulary index must be word_to_idx['<UNK>'] = -1 !
    mask_unk = -tf.cast(tf.greater(prob_masked, word_dropout_rate), tf.int32) 
    # All words that are kept should be the same as in captions_in, only change the ones in mask_unk to -1 
    captions_in = captions_in * (1 + mask_unk) + mask_unk 

    # Word embeddings
    w = _word_embedding(inputs=captions_in, dim_vocab=vocab_size, dim_emb=dim_emb)

    loss_list = []
    for t in range(T):

        with tf.variable_scope('lstm', reuse=(t!=0)):
            _, (c, h) = lstm_cell(inputs=w[:,t,:], state=[c, h])

        logits = _decode_lstm(h, dim_h, vocab_size, dropout_rate, reuse=(t!=0))
        loss_list.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t],logits=logits)*mask[:, t])
        all_hidden.append(tf.expand_dims( h*tf.expand_dims(h_mask[:, t], axis=1), axis=1) ) # store all masked hidden states

    loss = tf.stack(loss_list, axis=1)
    all_hidden = tf.concat(all_hidden, axis=1)
    return loss, all_hidden 

def _decode_lstm(h, dim_h, vocab_size, dropout_rate, reuse=False):
    """Compute logits for the predicted word.

    Args:
        h: Output hidden state from LSTM as input data.
        dim_h: Hidden state dimension.
        vocab_size: Number of words in vocabulary.
        dropout_rate: Dropout rate before linear layer.
        reuse: Indicates if variable scopes should be reused or to create new scopes.

    Returns:
        Logits for all words in vocabulary.

    """
    with tf.variable_scope('language_decoder', reuse=reuse):

        w_out = tf.get_variable('w_out', [dim_h, vocab_size], initializer=weight_initializer)
        b_out = tf.get_variable('b_out', [vocab_size], initializer=const_initializer)

        h1 = tf.nn.dropout(h, keep_prob=dropout_rate)
        out_logits = tf.matmul(h1, w_out) + b_out
        return out_logits


def _get_initial_lstm(features, dim_h):
    """Get LSTM network for text descriptions.

    Args:
        features: Input data
        dim_h: Hidden state dimension.

    Returns:
        Initial hidden and memory states for LSTM.

    """
    with tf.variable_scope('initial_lstm'):
        #features_mean = tf.reduce_mean(features, 1) # (N, L, D) -> (N, D)
        dim_f = features.get_shape().as_list()[-1]#tf.shape(features)[1] # Get input dimension

        w_h = tf.get_variable('w_h', [dim_f, dim_h], initializer=weight_initializer)
        b_h = tf.get_variable('b_h', [dim_h], initializer=const_initializer)
        #h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)
        h = tf.nn.tanh(tf.matmul(features, w_h) + b_h)

        w_c = tf.get_variable('w_c', [dim_f, dim_h], initializer=weight_initializer)
        b_c = tf.get_variable('b_c', [dim_h], initializer=const_initializer)
        #c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
        c = tf.nn.tanh(tf.matmul(features, w_c) + b_c)
        return c, h

def _word_embedding(inputs, dim_vocab, dim_emb, reuse=False):
    """Get word embedding from lookup table.

    Args:
        inputs: Input words.
        dim_vocab: Number of words in vocabulary.
        dim_emb: Word embedding dimension.
        reuse: Indicates if variable scopes should be reused or to create new scopes.

    Returns:
        Word embeddings.

    """
    with tf.variable_scope('word_embedding', reuse=reuse):
        w = tf.get_variable('w', [dim_vocab, dim_emb], initializer=emb_initializer)
        x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
        return x

def build_sampler(z, word_to_idx, dim_h, dim_emb, dropout_rate=1.0, max_len=20):
    """Get LSTM network for sampling text without supervision.

    Args:
        z: Input embedding for initializing the hidden states.
        word_to_idx: Vocabulary encoding words to index.
        dim_h: Hidden state dimension.
        dim_emb: Word embedding dimension.
        dropout_rate: Dropout rate before linear layer.
        max_len: Maximum number of words to generate/sample.

    Returns:
        Sampled text descriptions and all hidden states.

    """
    start_tag = word_to_idx['<START>']
    end_tag = word_to_idx['<END>']
    vocab_size = len(word_to_idx)

    # For storing all hidden states
    h_mask = tf.ones([])
    all_hidden = []

    c, h = _get_initial_lstm(features=z, dim_h=dim_h)
    h0 = h

    sampled_word_list = []
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=dim_h)

    for t in range(max_len):
        if t == 0:
            w = _word_embedding(inputs=tf.fill([tf.shape(z)[0]], start_tag), dim_vocab=vocab_size, dim_emb=dim_emb)
        else:
            w = _word_embedding(inputs=sampled_word, dim_vocab=vocab_size, dim_emb=dim_emb, reuse=True)

        with tf.variable_scope('lstm', reuse=(t!=0)):
            _, (c, h) = lstm_cell(inputs=w, state=[c, h])

        logits = _decode_lstm(h, dim_h, vocab_size, dropout_rate, reuse=(t!=0))
        sampled_word = tf.argmax(logits, 1)
        sampled_word_list.append(sampled_word)

        all_hidden.append(tf.expand_dims((h*h_mask), axis=1))
        # Update mask
        h_mask = tf.cast(tf.not_equal(sampled_word, end_tag), tf.float32)
        h_mask = tf.expand_dims(h_mask, axis=1)

    sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
    all_hidden = tf.concat(all_hidden, axis=1) 
    return sampled_captions, all_hidden

def language_encoder(captions, word_to_idx, T, dim_h, dim_emb, reuse=None):
    """Get LSTM network for encoding text descriptions.

    Args:
        captions: Text captions as input data.
        word_to_idx: Vocabulary encoding words to index.
        T: Number of steps for LSTM.
        dim_h: Hidden state dimension.
        dim_emb: Word embedding dimension.
        reuse: Indicates if variable scopes should be reused or to create new scopes.

    Returns:
        All hidden states and the final hidden state.

    """

    end_tag = word_to_idx['<END>']
    pad_tag = word_to_idx['<PAD>']
    vocab_size = len(word_to_idx)

    # Define in and out captions and mask
    captions_in = captions[:, :T]
    #captions_out = captions[:, 1:]
    mask = tf.to_float(tf.not_equal(captions[:, 1:], pad_tag))
    h_mask = tf.to_float(tf.equal(captions[:, 1:], end_tag))
    batch_size = tf.shape(captions)[0]
    
    # Initialize LSTM states
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=dim_h)
    (c, h) = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    #c = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    
    # Storing hidden states
    all_hidden = []

    # Word embeddings
    w = _word_embedding(inputs=captions_in, dim_vocab=vocab_size, dim_emb=dim_emb)

    for t in range(T):

        with tf.variable_scope('lstm', reuse=(t!=0)):
            _, (c, h) = lstm_cell(inputs=w[:,t,:], state=[c, h])
        
        # Store (masked) hidden state in all_hidden
        h_masked = tf.multiply(h, tf.expand_dims(mask[:, t], axis=-1) )#h*mask[:, t]
        all_hidden.append(tf.expand_dims(h_masked, axis=1) )

    all_hidden = tf.concat(all_hidden, axis=1) # stack along sequence dim
    h_ind = tf.where(h_mask) # Get indices of where the end tag is
    # Get the last hidden states that are valid according to their end tags using h_ind
    final_h = tf.gather_nd(all_hidden, h_ind) 
    return all_hidden, final_h