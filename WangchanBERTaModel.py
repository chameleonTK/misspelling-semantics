
import argparse
import math
import os
from functools import partial
import urllib.request
from tqdm import tqdm
from typing import Collection, Callable
from pathlib import Path
from sklearn import preprocessing
import pandas as pd
import numpy as np



import torch
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_constant_schedule, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    CamembertTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    XLMRobertaConfig,
    DataCollatorWithPadding,
    default_data_collator
)

from datasets import load_dataset, list_metrics, load_dataset, Dataset
# from thai2transformers.datasets import SequenceClassificationDataset
# from thai2transformers.metrics import classification_metrics, multilabel_classification_metrics
# from thai2transformers.finetuners import SequenceClassificationFinetuner
# from thai2transformers.auto import AutoModelForMultiLabelSequenceClassification
# from thai2transformers.tokenizers import (
#     ThaiRobertaTokenizer,
#     ThaiWordsNewmmTokenizer,
#     ThaiWordsSyllableTokenizer,
#     FakeSefrCutTokenizer,
# )
# from thai2transformers.utils import get_dict_val
# from thai2transformers.conf import Task
# from thai2transformers import preprocess

PUBLIC_MODEL = {
    # 'mbert': {
    #     'name': 'bert-base-multilingual-cased',
    #     'tokenizer': BertTokenizerFast,
    #     'config': BertConfig,
    # },
    # 'xlmr': {
    #     'name': 'xlm-roberta-base',
    #     'tokenizer': XLMRobertaTokenizerFast,
    #     'config': XLMRobertaConfig,
    # },
    # 'xlmr-large': {
    #     'name': 'xlm-roberta-large',
    #     'tokenizer': XLMRobertaTokenizerFast,
    #     'config': XLMRobertaConfig,
    # },
    'wangchanberta': {
        "name": "airesearch/wangchanberta-base-att-spm-uncased",
        "tokenizer": CamembertTokenizer
    }
}


## Custom Classes for Our Experiments
from transformers.modeling_roberta import RobertaEmbeddings
from torch import nn

class CustomRobertaEmbeddings(RobertaEmbeddings):

    def __init__(self, ref, config):
        super().__init__(config)
        self.word_embeddings = ref.word_embeddings

    def forward(self, input_ids, misp_ids, token_type_ids=None, position_ids=None, inputs_embeds=None):

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embeddings(input_ids)
        misp_embeds = self.word_embeddings(misp_ids)

        embeddings = ((inputs_embeds + misp_embeds)*0.5)
        return embeddings



from transformers.modeling_roberta import RobertaModel
from transformers.modeling_camembert import CamembertForSequenceClassification
from transformers.modeling_roberta import RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomSequenceClassification(CamembertForSequenceClassification):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, refmodel=None, mode="emb"):
        super().__init__(config)
        if refmodel is not None:
          config = refmodel.config
          # self.refmodel = refmodel
          self.num_labels = config.num_labels

          self.roberta = refmodel.roberta
          self.classifier = refmodel.classifier

          self.baseEmb = refmodel.roberta.embeddings
          self.newEmb = CustomRobertaEmbeddings(self.baseEmb, config)
          self.mode = mode

    def forward(self, *args, **kwargs):
        # del kwargs["misp_ids"]
        # return self.refmodel(**kwargs)

        return_dict = self.config.use_return_dict

        if self.mode == "emb":
            inputs_embeds = self.newEmb(kwargs["input_ids"], kwargs["misp_ids"])
            
            # doc: https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaModel.forward
            outputs = self.roberta(
                input_ids=None,
                attention_mask=kwargs["attention_mask"],
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
        else:

            inputs_embeds = self.newEmb(kwargs["input_ids"], kwargs["input_ids"])

            corr_outputs = self.roberta(
                input_ids=None,
                attention_mask=kwargs["attention_mask"],
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )

            inputs_embeds = self.newEmb(kwargs["misp_ids"], kwargs["misp_ids"])

            misp_outputs = self.roberta(
                input_ids=None,
                attention_mask=kwargs["attention_mask"],
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=inputs_embeds,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )

            if not return_dict:
                outputs = []
                for o1, o2 in zip(corr_outputs, misp_outputs):
                    if o1 is None or o2 is None:
                        outputs.append(None)
                    else:
                        outputs.append((o1+o2)/2)
                outputs = tuple(outputs)
            else:
                raise Exception("Not support for now")
            
            sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        labels = kwargs["labels"]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase

@dataclass
class CustomDataCollatorWithPadding:
  tokenizer: PreTrainedTokenizerBase
  padding: Union[bool, str] = True
  max_length: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None
  return_tensors: str = "pt"

  def __call__(self, features):
    _tmpfeat = []
    for f in features:
      _tmpfeat.append({
          "input_ids": f["input_ids"],
          "attention_mask": f["attention_mask"],
          "label": f["label"],
      })

    batch = self.tokenizer.pad(
        _tmpfeat,
        padding=self.padding,
        max_length=self.max_length,
        pad_to_multiple_of=self.pad_to_multiple_of,
        # return_tensors=self.return_tensors,
    )

    _tmpfeat = []
    for f in features:
      _tmpfeat.append({
          "input_ids": f["misp_ids"],
          "attention_mask": f["attention_mask"],
          "label": f["label"],
      })
      
    mispbatch = self.tokenizer.pad(
        _tmpfeat,
        padding=self.padding,
        max_length=self.max_length,
        pad_to_multiple_of=self.pad_to_multiple_of,
        # return_tensors=self.return_tensors,
    )

    # print(mispbatch["input_ids"])
    # print(tokenizer.convert_ids_to_tokens(batch["input_ids"][0]))
    # print(tokenizer.convert_ids_to_tokens(mispbatch["input_ids"][0]))
    batch["misp_ids"] = mispbatch["input_ids"]
    # assert(batch["misp_ids"].shape==batch["input_ids"].shape)
    # assert()
    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    if "label_ids" in batch:
        batch["labels"] = batch["label_ids"]
        del batch["label_ids"]
    return batch

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report

def classification_metrics(pred, pred_labs=False):
    labels = pred.label_ids
    preds = pred.predictions if pred_labs else pred.predictions.argmax(-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro")
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'nb_samples': len(labels)
    }

def init_trainer(model, train_dataset, val_dataset, warmup_steps, args, data_collator=default_data_collator): 
        
    training_args = TrainingArguments(
                        num_train_epochs=args.num_train_epochs,
                        per_device_train_batch_size=args.batch_size,
                        per_device_eval_batch_size=args.batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        learning_rate=args.learning_rate,
                        warmup_steps=warmup_steps,
                        weight_decay=args.weight_decay,
                        adam_epsilon=args.adam_epsilon,
                        max_grad_norm=args.max_grad_norm,
                        #checkpoint
                        output_dir=args.output_dir,
                        overwrite_output_dir=True,
                        #logs
                        logging_dir=args.log_dir,
                        logging_first_step=False,
                        logging_steps=args.logging_steps,
                        #eval
                        evaluation_strategy='epoch',
                        # save_strategy="epoch",
                        load_best_model_at_end=True,
                        #others
                        seed=args.seed,
                        fp16=args.fp16,
                        fp16_opt_level=args.fp16_opt_level,
                        dataloader_drop_last=False,
                        no_cuda=args.no_cuda,
                        metric_for_best_model=args.metric_for_best_model,
                        prediction_loss_only=False,
                        run_name=args.run_name
                    )
    compute_metrics_fn = classification_metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    return trainer, training_args

class WangchanBERTaModel:
    def __init__(self, model_name, num_labels):
        
        self.config = AutoConfig.from_pretrained(PUBLIC_MODEL[model_name]["name"], num_labels=num_labels);
        self.tokenizer = PUBLIC_MODEL[model_name]["tokenizer"].from_pretrained(PUBLIC_MODEL[model_name]["name"]);
        self.model = AutoModelForSequenceClassification.from_pretrained(PUBLIC_MODEL[model_name]["name"], config=self.config);
    
    def init_trainer(self, args, dataset_split, verbos=True):


        warmup_steps = math.ceil(len(dataset_split['train']) / args.batch_size * args.warmup_ratio * args.num_train_epochs)

        if verbos:
            print(f'\n[INFO] Number of train examples = {len(dataset_split["train"])}')
            print(f'[INFO] Number of batches per epoch (training set) = {math.ceil(len(dataset_split["train"]) / args.batch_size)}')

            print(f'[INFO] Warmup ratio = {args.warmup_ratio}')
            print(f'[INFO] Warmup steps = {warmup_steps}')
            print(f'[INFO] Learning rate: {args.learning_rate}')
            print(f'[INFO] Logging steps: {args.logging_steps}')
            print(f'[INFO] FP16 training: {args.fp16}\n')

            # if 'validation' in DATASET_METADATA[args.dataset_name]['split_names']:
            print(f'[INFO] Number of validation examples = {len(dataset_split["validation"])}')
            print(f'[INFO] Number of batches per epoch (validation set) = {math.ceil(len(dataset_split["validation"]))}')

        data_collator = CustomDataCollatorWithPadding(self.tokenizer,
                                                padding=True,
                                                pad_to_multiple_of=8 if args.fp16 else None)

        cusmodel = CustomSequenceClassification(self.model.config, self.model)
        return init_trainer(
                model=cusmodel,
                train_dataset=dataset_split['train'],
                val_dataset=dataset_split['validation'],
                warmup_steps=warmup_steps,
                args=args,
                data_collator=data_collator)
    
    def evaluate(self, trainer, dataset_split, mode="emb"):

        # assert(torch.equal(model.classifier.dense.weight, trainer.model.classifier.dense.weight))
        model = self.model.cuda()
        trainer.model = CustomSequenceClassification(model.config, model, mode=mode).cuda()
        trainer.model.eval();



        for split_name in dataset_split:
            if split_name.startswith("train"):
                continue

            p, label_ids, result = trainer.predict(test_dataset=dataset_split[split_name])
            print("")
            print(f'Evaluation on {split_name}:')    

            for key, value in result.items():
                print(f'{key} : {value:.4f}')
            
            print("*"*40)
            print()
