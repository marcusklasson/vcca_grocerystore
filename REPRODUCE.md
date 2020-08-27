# Reproduce results

A paper about this work is currently under submission.
If you are a reviewer, follow the steps below to reproduce classification
results and figures.

Start by installing the conda environment:
```
conda env create -f environment.yml
conda activate vcca_grocerystore
```

Download the Grocery Store dataset and DenseNet169 features:

[Github link to the Grocery Store dataset](https://github.com/marcusklasson/GroceryStoreDataset)

[Google Drive link to download DenseNet features](https://drive.google.com/file/d/1E_b6CR2ZaVyF60W9GUc7wT0RvNEqlQbr/view?usp=sharing)

The features should be stored in the same directory as where the preprocessed data is located.
The dataset can be stored wherever appropriate.

## Training the models

We trained all models with 10 random seeds (1-10) to report mean and standard deviations in the classification results.

Below are the instructions to reproduce the results in the submitted paper. 
We set ```--seed 1``` as an example.

Remember to store the downloaded DenseNet features in the directory ```./data/processed```.

### Models using natural image features only

VAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vae --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vae --seed 1
```

Autoencoder
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name ae --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name ae --seed 1
```


### Models using natural image features and class labels

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xy --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xy --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xy --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xy --seed 1
```


### Models using natural image features and iconic images

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xi --lambda_i 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xi --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xi --lambda_i 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xi --seed 1
```

VCCA-private
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_private_xi --lambda_i 10 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_private_xi --seed 1
```

### Models using natural image features, iconic images, and class labels

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xiy --lambda_i 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xiy --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xiy --lambda_i 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xiy --seed 1
```

VCCA-private
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_private_xiy --lambda_i 10 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_private_xiy --seed 1
```

### Models using natural image features and text descriptions

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 75
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xw --lambda_w 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xw --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 40
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xw --lambda_w 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xw --seed 1
```

VCCA-private
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 75
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_private_xw --lambda_w 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_private_xw --seed 1
```

### Models using natural image features, text descriptions, and class labels

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 75
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xwy --lambda_w 1000 --lambda_y 10 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xwy --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 75
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xwy --lambda_w 1000 --lambda_y 10 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xwy --seed 1
```

VCCA-private
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 50
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_private_xwy --lambda_w 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_private_xwy --seed 1
```

### Models using natural images, iconic images, and text descriptions

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 32
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xiw --lambda_i 1000 --lambda_w 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xiw --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 24
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xiw --lambda_i 1000 --lambda_w 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xiw --seed 1
```

### Models using natural images, iconic images, text descriptions, and class labels

VCCA
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 91
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name vcca_xiwy --lambda_i 1000 --lambda_w 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name vcca_xiwy --seed 1
```

SplitAE
```
python ./data/preprocessed_data.py --data_dir /path/to/GroceryStoreDataset/dataset --save_dir ./data/processed --caption_length 24
python train.py --data_path ./data/processed --model_dir ./saved_model --model_name splitae_xiwy --lambda_i 1000 --lambda_w 1000 --lambda_y 1000 --seed 1 --save_model 1
python test.py --data_path ./data/processed --model_dir ./saved_model --save_dir ./saved_images_and_metrics --model_name splitae_xiwy --seed 1
```

