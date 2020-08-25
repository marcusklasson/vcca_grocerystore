# VCCA for Grocery Store Dataset

Implementation of Variational Canonical Correlation Analysis (VCCA) for 
grocery item classification with the Grocery Store dataset. VCCA can
makes use of the web-scraped information in the dataset (i.e. iconic images and text 
descriptions) to learn better representations ofthe grocery items.

Note that the code is written in Tensorflow 1!

## Install conda environment
Install the conda environment by executing the following command in a terminal:
```
conda env create -f environment.yml
conda activate vcca_grocerystore
```

## Data

Download the Grocery Store dataset and store it wherever appropriate for you.

[Github link to the Grocery Store dataset](https://github.com/marcusklasson/GroceryStoreDataset)

### Preprocessing of data
Before training, we have to preprocess the data. 
The preprocessing includes fetching image paths and labels, and 
creating a vocabulary for the text descriptions. 
The default text description length is 36 words, but this can be changed
by 

```
python ./data/preprocess_data.py --data_path /path/to/GroceryStoreDataset \
	--save_dir /path/to/save/data
```

### Off-the-shelf features for the natural images
We have extracted off-the-shelf features from a pre-trained DenseNet169 network.
The features should be stored in the same directory as where the preprocessed data is located.

[Google Drive link to download DenseNet features](https://drive.google.com/file/d/1E_b6CR2ZaVyF60W9GUc7wT0RvNEqlQbr/view?usp=sharing) 


## Training

### Selecting a model
Start by selecting which model you wish to train:
* VCCA: ```vcca_xi, vcca_xiy, vcca_xw, vcca_xwy, vcca_xiw, vcca_xiwy, vcca_xy ```
* VCCA-private: ```vcca_private_xi, vcca_private_xiy, vcca_private_xw, vcca_private_xwy ```
* VAE: ```vae```
* SplitAutoencoder: ```splitae_xi, splitae_xiy, splitae_xw, splitae_xwy, splitae_xiw, splitae_xiwy, splitae_xy ```
* Autoencoder: ```ae```

The subscript indicates which data views from the dataset that are used during training:
* ```x```: Image features extracted from pre-trained DenseNet (download features above)
* ```i```: Iconic images of grocery items
* ```w```: Text descriptions of grocery items
* ```y```: Class labels of the natural images

The VAE and Autoencoder names do not use a subscript because they only use the image features ```x```.
If selecting a model without ```y```, then classification is performed by training a softmax classifier
on the latent representations of the model.

### Scaling weights for reconstruction losses
We can choose to scale the reconstruction losses for each view by passing a number to 
the following arguments to train.py:
* ```--lambda_x```: Scaling weight for image feature loss
* ```--lambda_i```: Scaling weight for iconic image loss 
* ```--lambda_i```: Scaling weight for text descriptions loss
* ```--lambda_y```: Scaling weight for class labels

The default values for all scaling weights is 1.

### Training the selected model
As an example, we train the model ```vcca_xiwy``` by executing:
```
python train.py --data_path /path/to/processed_data --model_name vcca_xiwy 
```
If we would like to change the scaling weights of the model, we pass value of 
the scaling weights as arguments by executing:
```
python train.py --data_path /path/to/processed_data --model_name vcca_xiwy \
				--lambda_i 1000 --lambda_w 1000 --lambda_y 1000
```

For saving the trained model, pass the argument ```--save_model 1```. 
You can also specify the directory where the model should be saved with the argument ```--model_dir /path/to/saved_model```.

## Test
You can load a trained model in the script ```test.py``` to
* compute the fine- and coarse-grained accuracy 
* plot the latent representations in 2D
* decode iconic images from natural images and compute metrics for the decoded images (if iconic image decoder was used in model) 

Run the script by executing:
```
python test.py --data_path /path/to/processed_data --model_dir /path/to/saved_model --model_name MODEL_NAME 
```
If the model used a softmax classifier, then pass the argument 
```--clf_dir /path/to/saved_classifier```

Metrics and images are saved in the directory passed as argument 
```--save_dir /path/to/saved_metrics_and_images```

## To-dos
- [ ] Should use inheritance for the models by writing a VAE base class that the VCCA models inherits from.