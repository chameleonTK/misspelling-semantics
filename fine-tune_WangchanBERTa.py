
from WangchanBERTaArgs import parser
import os
import numpy as np
import random
import torch
from WangchanBERTaModel import WangchanBERTaModel
from WangchanBERTaDataset import WangchanBERTaDataset

import warnings
warnings.filterwarnings("ignore")

def set_random_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)    
    np.random.seed(seed)
    np.random.RandomState(seed)

    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #seed all gpus    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  
    torch.backends.cudnn.benchmark = False

import sys
if __name__ == '__main__':
    DIR = "Models/WangchanBERTa"
    
    model_idx = 0
    model_name = "wangchanberta"

    if len(sys.argv) >= 2:
        DIR = sys.argv[1]
    
    if len(sys.argv) >= 3:
        model_idx = int(sys.argv[2])

    # model_name = "xlmr"
    # model_name = "mbert"
    args_params = f"{model_name} wisesight_sentiment {DIR}/Outputs/ {DIR}/Logs/ --batch_size 32 --seed {model_idx} --num_train_epochs 10"
    args = parser.parse_args(args_params.split())
    set_random_seed(args.seed)
    num_labels = 3
    MODEL = WangchanBERTaModel(model_name, num_labels)


    print('\n[INFO] Preprocess and tokenizing texts in datasets')
    max_length = args.max_length if args.max_length else MODEL.config.max_position_embeddings
    print(f'[INFO] max_length = {max_length} \n')
    DATASETS = WangchanBERTaDataset("./Datasets/WisesightSentiment", MODEL.tokenizer)
    dataset_split = DATASETS.preprocess()

    # Model Training 
    trainer, training_args = MODEL.init_trainer(args, dataset_split)

    print('[INFO] TrainingArguments:')
    print(training_args)
    print('\n')


    print('\nBegin model finetuning.')
    trainer.train()
    print('Done.\n')

    trainer.save_model(f"{DIR}/fined-tune-{model_name}")

    # Evaluation
    MODEL.evaluate(trainer, dataset_split)

    print("====================================================")

    MODEL.evaluate(trainer, dataset_split, mode="cls")

