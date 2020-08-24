
def get_model(args, dataset):
    """Get model for experiment.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        dataset: Dataset with information passed as arguments to models.

    Returns:
        The model for the experiment. 

    """

    word_to_idx = dataset['word_to_idx']
    captions = dataset['captions']
    dim_w = captions.shape[-1] - 1 # nbr of words in caption + 1!
    dim_x = dataset['features'].shape[-1]
    n_classes = dataset['n_classes']
    dim_iconic_image = [args.iconic_img_size, args.iconic_img_size, 3]
    
    if args.model_name == 'vae':
        from models.vae import VAE
        model = VAE(dim_x=dim_x, dim_z=args.z_dim, lambda_x=args.lambda_x,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,
            use_batchnorm=args.use_batchnorm)

    elif args.model_name == 'vcca_xy':
        from models.vcca.vcca_xy import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes,
            lambda_x=args.lambda_x, lambda_y=args.lambda_y, n_layers_classifier=args.n_layers_classifier,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, use_batchnorm=args.use_batchnorm)

    elif args.model_name == 'vcca_xw':
        from models.vcca.vcca_xw import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_w=dim_w,
            lambda_x=args.lambda_x, lambda_w=args.lambda_w, word_to_idx=word_to_idx,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,)

    elif args.model_name == 'vcca_private_xw':
        from models.vcca_private.vcca_private_xw import VCCA_private
        model = VCCA_private(dim_x=dim_x, dim_z=args.z_dim, dim_w=dim_w,
            lambda_x=args.lambda_x, lambda_w=args.lambda_w, word_to_idx=word_to_idx,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,)

    elif args.model_name == 'vcca_xwy':
        from models.vcca.vcca_xwy import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, dim_w=dim_w,
            lambda_x=args.lambda_x, lambda_w=args.lambda_w, lambda_y=args.lambda_y, word_to_idx=word_to_idx,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, n_layers_classifier=args.n_layers_classifier)

    elif args.model_name == 'vcca_private_xwy':
        from models.vcca_private.vcca_private_xwy import VCCA_private
        model = VCCA_private(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, dim_w=dim_w,
            lambda_x=args.lambda_x, lambda_w=args.lambda_w, lambda_y=args.lambda_y, word_to_idx=word_to_idx,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, n_layers_classifier=args.n_layers_classifier)

    elif args.model_name == 'vcca_xi':
        from models.vcca.vcca_xi import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_i=dim_iconic_image, lambda_x=args.lambda_x, lambda_i=args.lambda_i,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)

    elif args.model_name == 'vcca_private_xi':
        from models.vcca_private.vcca_private_xi import VCCA_private
        model = VCCA_private(dim_x=dim_x, dim_z=args.z_dim, dim_i=dim_iconic_image, lambda_x=args.lambda_x, lambda_i=args.lambda_i,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)

    elif args.model_name == 'vcca_xiy':
        from models.vcca.vcca_xiy import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, dim_i=dim_iconic_image,
            lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_y=args.lambda_y, 
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, n_layers_classifier=args.n_layers_classifier) 

    elif args.model_name == 'vcca_private_xiy':
        from models.vcca_private.vcca_private_xiy import VCCA_private
        model = VCCA_private(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, dim_i=dim_iconic_image,
            lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_y=args.lambda_y, 
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, n_layers_classifier=args.n_layers_classifier) 

    elif args.model_name == 'vcca_xiw':
        from models.vcca.vcca_xiw import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_y=dim_iconic_image, dim_w=dim_w,
             word_to_idx=word_to_idx, lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_w=args.lambda_w,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)     

    elif args.model_name == 'vcca_xiwy':
        from models.vcca.vcca_xiwy import VCCA
        model = VCCA(dim_x=dim_x, dim_z=args.z_dim, dim_i=dim_iconic_image, dim_w=dim_w,
             word_to_idx=word_to_idx, lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_w=args.lambda_w,
             lambda_y=args.lambda_y, dim_labels=n_classes, n_layers_classifier=args.n_layers_classifier,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)              

    elif args.model_name == 'ae':
        from models.ae import Autoencoder
        model = Autoencoder(dim_x=dim_x, dim_z=args.z_dim, lambda_x=args.lambda_x,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,
            use_batchnorm=args.use_batchnorm)

    elif args.model_name == 'splitae_xy':
        from models.splitae.splitae_xy import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, lambda_x=args.lambda_x, lambda_y=args.lambda_y,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,
            use_batchnorm=args.use_batchnorm, n_layers_classifier=args.n_layers_classifier)  

    elif args.model_name == 'splitae_xw':
        from models.splitae.splitae_xw import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_w=dim_w,
            lambda_x=args.lambda_x, lambda_w=args.lambda_w, word_to_idx=word_to_idx,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder,)

    elif args.model_name == 'splitae_xwy':
        from models.splitae.splitae_xwy import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, dim_w=dim_w,
            lambda_x=args.lambda_x, lambda_w=args.lambda_w, lambda_y=args.lambda_y, word_to_idx=word_to_idx,
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, n_layers_classifier=args.n_layers_classifier)

    elif args.model_name == 'splitae_xi':
        from models.splitae.splitae_xi import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_i=dim_iconic_image, lambda_x=args.lambda_x, lambda_i=args.lambda_i,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)

    elif args.model_name == 'splitae_xiy':
        from models.splitae.splitae_xiy import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_labels=n_classes, dim_i=dim_iconic_image,
            lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_y=args.lambda_y, 
            n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder, n_layers_classifier=args.n_layers_classifier)  

    elif args.model_name == 'splitae_xiw':
        from models.splitae.splitae_xiw import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_i=dim_iconic_image, dim_w=dim_w,
             word_to_idx=word_to_idx, lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_w=args.lambda_w,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)     

    elif args.model_name == 'splitae_xiwy':
        from models.splitae.splitae_xiwy import SplitAE
        model = SplitAE(dim_x=dim_x, dim_z=args.z_dim, dim_i=dim_iconic_image, dim_w=dim_w,
             word_to_idx=word_to_idx, lambda_x=args.lambda_x, lambda_i=args.lambda_i, lambda_w=args.lambda_w,
             lambda_y=args.lambda_y, dim_labels=n_classes, n_layers_classifier=args.n_layers_classifier,
             n_layers_encoder=args.n_layers_encoder, n_layers_decoder=args.n_layers_decoder)  

    else:
        ValueError('Unknown model: %s'.format(args.model_name))
    return model