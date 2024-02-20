# federated domain generalization

## Available methods
* FedAvg
* DeepCoral
* FedDG
* FedADG
* FedSR


# Environment preparation
```
conda create  --name <env> --file requirements.txt
```

## Prepare Datasets
All datasets derived from [Wilds](https://wilds.stanford.edu/) Datasets. We also implement [femnist](https://leaf.cmu.edu/) and [PACS](https://arxiv.org/abs/2007.01434) datasets.

### Preparing metadata.csv and RELEASE_v1.0.txt
For PACS and FEMNIST dataset, please put 
```
resources/femnist_v1.0/* 
```
and 
```
resources/pacs_v1.0/* 
```
into your dataset directory.

### Preparing fourier transformation
Some methods require fourier transformation. To accelerate training, we should prepare the transformation data in advance. Please first load the scripts in the scripts path. Note: Please config the root_path in the script.

## Run Experiments
To run the experiments, simply prepare your config file $config_path, and run
```
python main.py --config_file $config_path
```
For example, to run fedavg-erm with centralized learning on iwildcam, run
```
python main.py --config_file ./config/ERM/iwildcam/centralized.json
```

