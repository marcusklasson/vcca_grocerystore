
def get_training_function(model_name):
    """ Return function for executing training epoch.

    Args: 
        model_name: Name of model (string).

    Returns:
        A function that is used in the training loop
        for executing training/validation epoch.
    """
    
    if model_name == 'vae':
        from models.vae import run_training_epoch as func

    elif model_name == 'vcca_xy':
        from models.vcca.vcca_xy import run_training_epoch as func

    elif model_name == 'vcca_xw':
        from models.vcca.vcca_xw import run_training_epoch as func

    elif model_name == 'vcca_private_xw':
        from models.vcca_private.vcca_private_xw import run_training_epoch as func

    elif model_name == 'vcca_xwy':
        from models.vcca.vcca_xwy import run_training_epoch as func

    elif model_name == 'vcca_private_xwy':
        from models.vcca_private.vcca_private_xwy import run_training_epoch as func

    elif model_name == 'vcca_xi':
        from models.vcca.vcca_xi import run_training_epoch as func

    elif model_name == 'vcca_private_xi':
        from models.vcca_private.vcca_private_xi import run_training_epoch as func

    elif model_name == 'vcca_xiy':
        from models.vcca.vcca_xiy import run_training_epoch as func

    elif model_name == 'vcca_private_xiy':
        from models.vcca_private.vcca_private_xiy import run_training_epoch as func

    elif model_name == 'vcca_xiw':
        from models.vcca.vcca_xiw import run_training_epoch as func

    elif model_name == 'vcca_xiwy':
        from models.vcca.vcca_xiwy import run_training_epoch as func

    elif model_name == 'ae':
        from models.ae import run_training_epoch as func

    elif model_name == 'splitae_xy':
        from models.splitae.splitae_xy import run_training_epoch as func

    elif model_name == 'splitae_xw':
        from models.splitae.splitae_xw import run_training_epoch as func

    elif model_name == 'splitae_xwy':
        from models.splitae.splitae_xwy import run_training_epoch as func

    elif model_name == 'splitae_xi':
        from models.splitae.splitae_xi import run_training_epoch as func

    elif model_name == 'splitae_xiy':
        from models.splitae.splitae_xiy import run_training_epoch as func

    elif model_name == 'splitae_xiw':
        from models.splitae.splitae_xiw import run_training_epoch as func

    elif model_name == 'splitae_xiwy':
        from models.splitae.splitae_xiwy import run_training_epoch as func

    else:
        ValueError('Unknown model: %s'.format(model_name))

    return func