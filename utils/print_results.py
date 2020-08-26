
import time 

def print_results(args, results, mode='train', epoch=None, t_start=None):
    """ Prints metrics from epoch for the model. 

    Args:
        args: Arguments from parser in train_grocerystore.py.
        results: The evaluated metrics, such as loss and accuracy.
        mode: Training or validation epoch.
        epoch: The epoch index.
        t_start: Time stamp when epoch started.

    """
    if mode=='train':
        if args.model_name == 'vae':
            print('Epoch {:d}: ELBO: {:.3f}, X Recon. Loss: {:.3f}, KL: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['kl_loss'], (time.time() - t_start))) 

        elif args.model_name == 'vcca_xy':
            print('Epoch {:d}: ELBO: {:.3f}, X Recon. Loss: {:.3f}, KL: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(epoch, results['total_loss'],
                 results['x_loss'], results['kl_loss'], results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'vcca_xw':
            print('Epoch {:d}: ELBO: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, KL: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['w_loss'], results['kl_loss'], (time.time() - t_start)))

        elif args.model_name == 'vcca_private_xw':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},'
                ' KL_z: {:.3f}, KL_ux: {:.3f}, KL_uw: {:.3f} s/epoch: {:.3e}'.format(epoch,
                 results['total_loss'], results['x_loss'], results['w_loss'], results['kl_loss_z'],
                 results['kl_loss_ux'], results['kl_loss_uw'], (time.time() - t_start)))

        elif args.model_name == 'vcca_xwy':
            print('Epoch {:d}: ELBO: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, KL: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(epoch, results['total_loss'],
                 results['x_loss'], results['w_loss'], results['kl_loss'], results['clf_loss'], results['accuracy'],
                 (time.time() - t_start)))

        elif args.model_name == 'vcca_private_xwy':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},'
                ' KL_z: {:.3f}, KL_ux: {:.3f}, KL_uw: {:.3f}, Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['w_loss'], results['kl_loss_z'],
                results['kl_loss_ux'], results['kl_loss_uw'],results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'vcca_xi':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, KL: {:.3f} s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['i_loss'], results['kl_loss'], (time.time() - t_start)))

        elif args.model_name == 'vcca_private_xi':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f},'
                ' KL_z: {:.3f}, KL_ux: {:.3f}, KL_ui: {:.3f} s/epoch: {:.3e}'.format(epoch,
                 results['total_loss'], results['x_loss'], results['i_loss'], results['kl_loss_z'],
                 results['kl_loss_ux'], results['kl_loss_ui'], (time.time() - t_start)))

        elif args.model_name == 'vcca_xiy':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f},'
                ' KL: {:.3f}, Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['i_loss'], results['kl_loss'],
                results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'vcca_private_xiy':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f},'
                ' KL_z: {:.3f}, KL_ux: {:.3f}, KL_ui: {:.3f}, Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['i_loss'], results['kl_loss_z'],
                results['kl_loss_ux'], results['kl_loss_ui'],results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'vcca_xiw':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},'
                ' KL: {:.3f} s/epoch: {:.3e}'.format(epoch,
                results['total_loss'], results['x_loss'], results['i_loss'], results['w_loss'],
                results['kl_loss'], (time.time() - t_start)))

        elif args.model_name == 'vcca_xiwy':
            print('Epoch {:d}: ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},'
                ' KL: {:.3f}, Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['i_loss'], results['w_loss'], results['kl_loss'],
                results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'ae':
            print('Epoch {:d}: Total loss: {:.3f}, s/epoch: {:.3e}'.format(epoch,
             results['total_loss'], (time.time() - t_start)))

        elif args.model_name == 'splitae_xy':
            print('Epoch {:d}: Total loss: {:.3f}, X Recon. Loss: {:.3f}, ' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(epoch, results['total_loss'],
                 results['x_loss'], results['clf_loss'], results['accuracy'], (time.time() - t_start)))  

        elif args.model_name == 'splitae_xw':
            print('Epoch {:d}: Total loss: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['w_loss'], (time.time() - t_start)))  

        elif args.model_name == 'splitae_xwy':
            print('Epoch {:d}: Total loss: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(epoch, results['total_loss'],
                 results['x_loss'], results['w_loss'], results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'splitae_xi':
            print('Epoch {:d}: Total loss: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['i_loss'], (time.time() - t_start)))

        elif args.model_name == 'splitae_xiy':
            print('Epoch {:d}: Total loss: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f},'
                'Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(epoch,
                results['total_loss'], results['x_loss'], results['i_loss'],
                results['clf_loss'], results['accuracy'], (time.time() - t_start)))

        elif args.model_name == 'splitae_xiw':
            print('Epoch {:d}: Total loss: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f},'
                ' W X-Entropy Loss: {:.3f}, s/epoch: {:.3e}'.format(epoch,
                results['total_loss'], results['x_loss'], results['i_loss'], results['w_loss'], (time.time() - t_start)))

        elif args.model_name == 'splitae_xiwy':
            print('Epoch {:d}: Total loss: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},'
                ' Classifier loss: {:.3f}, Accuracy: {:.3f}, s/epoch: {:.3e}'.format(
                epoch, results['total_loss'], results['x_loss'], results['i_loss'], results['w_loss'],
                results['clf_loss'], results['accuracy'], (time.time() - t_start)))


    elif mode == 'val':
        if args.model_name == 'vae':
            print('Validation: ELBO: {:.3f}, X Recon. Loss: {:.3f}, KL: {:.3f}'.format(
                results['total_loss'], results['x_loss'], results['kl_loss']))

        elif args.model_name == 'vcca_xy':
            print('Validation: ELBO: {:.3f}, X Recon. Loss: {:.3f}, KL: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['kl_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'vcca_xw':
            print('Validation: ELBO: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, KL: {:.3f}'.format(
                results['total_loss'], results['x_loss'], results['w_loss'], results['kl_loss']))

        elif args.model_name == 'vcca_xwy':
            print('Validation: ELBO: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, KL: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['w_loss'], results['kl_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'vcca_xi':
            print('Validation ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, KL: {:.3f} '.format(
                results['total_loss'], results['x_loss'], results['i_loss'], results['kl_loss']))

        elif args.model_name == 'vcca_xiy':
            print('Validation: ELBO: {:.3f}, X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, KL: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['i_loss'], results['kl_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'vcca_xiw':
            print('Validation ELBO: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, KL: {:.3f} '.format(
                results['total_loss'], results['x_loss'], results['i_loss'], results['w_loss'], results['kl_loss']))

        elif args.model_name == 'vcca_xiwy':
            print('Validation: ELBO: {:.3f}, X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}, KL: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['i_loss'], results['w_loss'], results['kl_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'ae':
            print('Validation: Total loss: {:.3f},'.format(results['total_loss']))

        elif args.model_name == 'splitae_xy':
            print('Validation: Total loss: {:.3f}, X Recon. Loss: {:.3f}, Classifier loss: {:.3f}, Accuracy: {:.3f}'.format(
             results['total_loss'], results['x_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'splitae_xw':
            print('Validation: Total loss: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}'.format(
                results['total_loss'], results['x_loss'], results['w_loss']))

        elif args.model_name == 'splitae_xwy':
            print('Validation: Total loss: {:.3f}, X Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['w_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'splitae_xi':
            print('Validation Total loss: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}'.format(
                results['total_loss'], results['x_loss'], results['i_loss']))

        elif args.model_name == 'splitae_xiy':
            print('Validation: Total loss: {:.3f}, X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['i_loss'], results['clf_loss'], results['accuracy']))

        elif args.model_name == 'splitae_xiw':
            print('Validation Total loss: {:.3f} X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f}'.format(
                results['total_loss'], results['x_loss'], results['i_loss'], results['w_loss']))

        elif args.model_name == 'splitae_xiwy':
            print('Validation: Total loss: {:.3f}, X Recon. Loss: {:.3f}, I Recon. Loss: {:.3f}, W X-Entropy Loss: {:.3f},' 
                ' Classifier loss: {:.3f}, Accuracy: {:.3f} '.format(results['total_loss'],
                 results['x_loss'], results['i_loss'], results['w_loss'], results['clf_loss'], results['accuracy']))

    else:
        ValueError('Unknown printing mode: %s'.format(mode))

def write_accuracy_to_file(args, accuracy, accuracy_coarse):
    """ Writes evaluated accuracies and the scaling weights
        for each data view to file.

    Args:
        args: Arguments from parser in train_grocerystore.py.
        accuracy: Fine-grained accuracy.
        accuracy_coarse: Coarse-grained accuracy.

    """
    model_name = args.model_name
    with open(args.save_file, 'a') as file:
        if model_name == 'vae' or model_name == 'ae':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x))

        elif model_name == 'vcca_xy' or model_name == 'splitae_xy':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_y  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_y))

        elif model_name == 'vcca_xw' or model_name == 'splitae_xw' or model_name == 'vcca_private_xw':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_w  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_w))

        elif model_name == 'vcca_xwy' or model_name == 'splitae_xwy' or model_name == 'vcca_private_xwy':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_w  Lambda_y  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_w, args.lambda_y))

        elif model_name == 'vcca_xi' or model_name == 'splitae_xi' or model_name == 'vcca_private_xi':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_i  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_i))

        elif model_name == 'vcca_xiy' or model_name == 'splitae_xiy' or model_name == 'vcca_private_xiy':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_i  Lambda_y  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_i, args.lambda_y))

        elif model_name == 'vcca_xiw' or model_name == 'splitae_xiw':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_i  Lambda_w  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_i, args.lambda_w))

        elif model_name == 'vcca_xiwy' or model_name == 'splitae_xiwy':
            file.write("Seed  Accuracy  Coarse Accuracy    Lambda_x  Lambda_i  Lambda_w  Lambda_y  \n")
            file.write("{:d}  {:.3f}    {:.3f}      {:.1f}    {:.1f}    {:.1f}    {:.1f} \n".format(
                args.seed, accuracy, accuracy_coarse, args.lambda_x, args.lambda_i, args.lambda_w, args.lambda_y))

        else:
            ValueError('Unknown model: %s'.format(model_name))
