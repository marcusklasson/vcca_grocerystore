<h1> VCCA for Grocery Store Dataset </h1>

This is the official implementation of our paper 
[Using Variational Multi-view Learning for Classification of Grocery Items](https://www.sciencedirect.com/science/article/pii/S2666389920301914) 
that was published in the Cell Press journal Patterns. 
This journal paper is an an extension of our [WACV 2019 paper](https://arxiv.org/abs/1901.00711).
The repository includes implementation of Variational Canonical Correlation Analysis (VCCA) for 
grocery item classification with the Grocery Store dataset. VCCA can
make use of the web-scraped information in the dataset (i.e. iconic images and text 
descriptions) to learn better representations of the grocery items.

# Usage

Follow the instructions below to perform experiments with the
implemented models. Note that the code is written in Tensorflow 1!

See [REPRODUCE](./REPRODUCE.md) for instructions to reproducing the results in the paper.

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
python ./data/preprocess_data.py --data_dir /path/to/GroceryStoreDataset/dataset \
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

The VAE and Autoencoder model names do not use a subscript because they only use the image features ```x```.
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
python train.py --data_path /path/to/processed_data --model_name vcca_xiwy --lambda_i 1000 --lambda_w 1000 --lambda_y 1000
```
The file ```clf_metrics.txt``` includes the fine- and coarse-grained accuracies predicted
by the used classifier.

For saving the trained model, pass the argument ```--save_model 1```. 
You can also specify the directory where the model should be saved with the argument ```--model_dir /path/to/saved_model```.
If the softmax classifier is used, then it is stored in ```/path/to/saved_model/saved_classifier```.

## Test
You can load a trained model in the script ```test.py``` to
* compute the fine- and coarse-grained accuracy 
* plot the latent representations in 2D
* decode iconic images from natural images and compute metrics for the decoded images (if iconic image decoder was used in model) 

Run the script by executing:
```
python test.py --data_path /path/to/processed_data --model_dir /path/to/saved_model --model_name MODEL_NAME 
```
The metrics and images are saved in the directory ```saved_images_and_metrics``` by default.
The directory can be passed as argument with ```--save_dir /path/to/new_name_for_saved_metrics_and_images```

### Latent representation in 2D using PCA
<p align="center">
  <img src="/figures/latent_representations/pca_latents_z_vae.png" width="300" title="hover text">
  <img src="/figures/latent_representations/pca_latents_z_vcca_xiwy.png" width="300" title="hover text">
  
  **Fig 1:** Latent representations plotted with the corresponding iconic images of models ```vae``` and ```vcca_xiwy```
in the left and right figure respectively.

</p>

### Decoded iconic images
<p align="center">
  <img src="/figures/natural_images/Mango_002_image477.jpg" width="100" title="hover text">
  <img src="/figures/true_iconic_images/Mango_Iconic.jpg" width="100" title="hover text">
  <img src="/figures/decoded_iconic_images/vcca_xiwy/mango_image477.png" width="100" title="hover text">
  <img src="/figures/natural_images/Royal-Gala_055_image266.jpg" width="100" title="hover text">
  <img src="/figures/true_iconic_images/Royal-Gala-Apple_Iconic.jpg" width="100" title="hover text">
  <img src="/figures/decoded_iconic_images/vcca_xiwy/royal_gala_image266.png" width="100" title="hover text">
</p>
<p align="center">
  <img src="/figures/natural_images/Orange-Bell-Pepper_008_image2191.jpg" width="100" title="hover text">
  <img src="/figures/true_iconic_images/Orange-Bell-Pepper_Iconic.jpg" width="100" title="hover text">
  <img src="/figures/decoded_iconic_images/vcca_xiwy/orange_bell_pepper_image2191.png" width="100" title="hover text">
  <img src="/figures/natural_images/Arla-Ecological-Sour-Cream_005_image1565.jpg" width="100" title="hover text">
  <img src="/figures/true_iconic_images/Arla-Ecological-Sour-Cream_Iconic.jpg" width="100" title="hover text">
  <img src="/figures/decoded_iconic_images/vcca_xiwy/arla_eco_sourcream_image1565.png" width="100" title="hover text">
  
  **Fig. 2:** Four examples of decoded iconic images from model ```vcca_xiwy``` by encoding the natural image and
  decoding the retrieved latent representation through the iconic image decoder. The images are structured
in the following order: 1) natural image, 2) true iconic image, 3) decoded iconic image.

</p>

## Citation
If you use this code or the Grocery Store dataset for your research, please cite our papers:

```
@article{klasson2020using,
  title={Using Variational Multi-view Learning for Classification of Grocery Items},
  author={Klasson, Marcus and Zhang, Cheng and Kjellstr{\"o}m, Hedvig},
  journal={Patterns},
  volume={1},
  number={8},
  pages={100143},
  year={2020},
  publisher={Elsevier}
}
```
```
@inproceedings{klasson2019hierarchical,
  title={A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels},
  author={Klasson, Marcus and Zhang, Cheng and Kjellstr{\"o}m, Hedvig},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2019}
}
```

## Acknowledgement
This research was funded by [Stiftelsen Promobilia](https://www.promobilia.se/) in Stockholm, Sweden.
