# VCCA for Grocery Store Dataset


## Install conda environment
Install the conda environment by executing the following command in a terminal:
```
conda env create -f environment.yml
conda activate vcca_grocerystore
```

## Data
Download the [Grocery Store dataset](https://github.com/marcusklasson/GroceryStoreDataset).

Download [DenseNet169 features](https://drive.google.com/file/d/1E_b6CR2ZaVyF60W9GUc7wT0RvNEqlQbr/view?usp=sharing) 
that can be used for training.

Before training, we have to preprocess the data using pandas. 
The preprocessing includes fetching image paths and labels, and 
creating a vocabulary for the text descriptions.

```
python ./data/preprocess_data.py --data_path /path/to/GroceryStoreDataset \
	--save_dir /path/to/save/data_splits \
	--caption_length text_description_length
```

## Training


```
python train.py --model_name model_name --data_path /path/to/processed_data
```