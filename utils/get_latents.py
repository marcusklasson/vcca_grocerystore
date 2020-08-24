
import numpy as np
import tensorflow as tf

from data.import_data import load_captions, load_iconic_images

def add_latents_to_dataset(args, sess, model, data):
    """ Get latent representations from model.
    
    Args:
        args: Arguments from parser in train_grocerystore.py.
        sess: Tensorflow session.
        model: Model used during epoch. 
        data: Data used during epoch.
       
    Returns:
        Data dictionary filled with latent representations.

    """

    latents = sess.run(model.latent_rep, feed_dict={model.x: data['features']} )
    data['latents'] = latents

    if args.use_private:
        latents_ux = sess.run(model.latent_rep_ux, feed_dict={model.x: data['features']} )
        data['latents_ux'] = latents_ux
        if args.use_text:
            all_captions = load_captions(data['captions'], data['labels'])
            latents_uw = sess.run(model.latent_rep_uw,
                                feed_dict={model.captions: all_captions})
            data['latents_uw'] = latents_uw
        if args.use_iconic:
            batch_size = args.batch_size
            n_examples = len(data['iconic_image_paths'])
            n_batches = int(np.ceil(n_examples/batch_size))
            latents_ui = np.zeros([n_examples, args.z_dim])
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                if end > n_examples:
                    end = n_examples
                iconic_images = load_iconic_images(data['iconic_image_paths'][start:end])
                latents_ui[start:end] = sess.run(model.latent_rep_ui,
                                feed_dict={model.iconic_images: iconic_images})
            data['latents_ui'] = latents_ui
    return data