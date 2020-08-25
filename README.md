# VCCA for Grocery Store Dataset


## Install conda environment
Install the conda environment by executing the following command in a terminal:
```
conda env create -f environment.yml
conda activate vcca_grocerystore
```

## Data

Download the Grocery Store dataset and store it wherever appropriate for you.

[Github link to the Grocery Store dataset](https://github.com/marcusklasson/GroceryStoreDataset).

Before training, we have to preprocess the data. 
The preprocessing includes fetching image paths and labels, and 
creating a vocabulary for the text descriptions. 
Remember to specifiy a 

```
python ./data/preprocess_data.py --data_path /path/to/GroceryStoreDataset \
	--save_dir /path/to/save/data
```

We have extracted off-the-shelf features from a pre-trained DenseNet169 network.
The features should be stored in the same directory as where the preprocessed data is located.

[Google Drive link to download DenseNet features](https://drive.google.com/file/d/1E_b6CR2ZaVyF60W9GUc7wT0RvNEqlQbr/view?usp=sharing) 


## Training


```
python train.py --model_name model_name --data_path /path/to/processed_data
```

## Test
If you want to 
* compute the fine- and coarse-grained accuracy 
* visualize the latent representations
* decode iconic images from natural images and compute metrics for the decoded images (if iconic image decoder was used in model) 

Then run the following command:
```
python test.py --model_name model_name --data_path /path/to/processed_data --model_dir
```