# Focus Where It Counts: A Salience-Driven Vision-Language Model for Low-Vision Assistance



## Create conda environment 
- conda create --name sllava python=3.9
- conda activate sllava
- conda env create -f environment.yml

## Dataset Access
Download dataset: <a href="https://drive.google.com/file/d/13YiuT3m2K8EP31HJkA9Gmx26AGyBTqpO/view?usp=sharing" target="_blank">Dataset</a>

## Model Operations
- Finetuning: Execute scripts in `/scripts/finetune/` (please set report to wandb if you would like to use it)
- Evaluation: Run `FT_benchmark.py`

## Evaluation Metrics
- Please collect the converstaion json file
- Get the final evaluation metrics by running `json_eval.py`


## Real World Experiment
- coming soon
