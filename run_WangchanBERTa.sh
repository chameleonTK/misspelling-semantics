#!/bin/bash

python fine-tune_WangchanBERTa.py Models/WangchanBERTa1 1 > Results/fine-tune_WangchanBERTa_1.out
rm -rf Models/WangchanBERTa1/Outputs
python fine-tune_WangchanBERTa_MST.py Models/WangchanBERTa1 1 > Results/fine-tune_WangchanBERTa_MST_1.out
rm -rf Models/WangchanBERTa1/Outputs
python few-shot_WangchanBERTa.py Models/WangchanBERTa1 1 > Results/few-shot_WangchanBERTa_1.out
rm -rf Models/WangchanBERTa1/Outputs
python few-shot_WangchanBERTa_MST.py Models/WangchanBERTa1 1 > Results/few-shot_WangchanBERTa_MST_1.out
rm -rf Models/WangchanBERTa1/Outputs

python fine-tune_WangchanBERTa.py Models/WangchanBERTa2 2 > Results/fine-tune_WangchanBERTa_2.out
rm -rf Models/WangchanBERTa2/Outputs
python fine-tune_WangchanBERTa_MST.py Models/WangchanBERTa2 2 > Results/fine-tune_WangchanBERTa_MST_2.out
rm -rf Models/WangchanBERTa2/Outputs
python few-shot_WangchanBERTa.py Models/WangchanBERTa2 2 > Results/few-shot_WangchanBERTa_2.out
rm -rf Models/WangchanBERTa2/Outputs
python few-shot_WangchanBERTa_MST.py Models/WangchanBERTa2 2 > Results/few-shot_WangchanBERTa_MST_2.out
rm -rf Models/WangchanBERTa2/Outputs

python fine-tune_WangchanBERTa.py Models/WangchanBERTa3 3 > Results/fine-tune_WangchanBERTa_3.out
rm -rf Models/WangchanBERTa3/Outputs
python fine-tune_WangchanBERTa_MST.py Models/WangchanBERTa3 3 > Results/fine-tune_WangchanBERTa_MST_3.out
rm -rf Models/WangchanBERTa3/Outputs
python few-shot_WangchanBERTa.py Models/WangchanBERTa3 3 > Results/few-shot_WangchanBERTa_3.out
rm -rf Models/WangchanBERTa3/Outputs
python few-shot_WangchanBERTa_MST.py Models/WangchanBERTa3 3 > Results/few-shot_WangchanBERTa_MST_3.out
rm -rf Models/WangchanBERTa3/Outputs

python fine-tune_WangchanBERTa.py Models/WangchanBERTa4 4 > Results/fine-tune_WangchanBERTa_4.out
rm -rf Models/WangchanBERTa4/Outputs
python fine-tune_WangchanBERTa_MST.py Models/WangchanBERTa4 4 > Results/fine-tune_WangchanBERTa_MST_4.out
rm -rf Models/WangchanBERTa4/Outputs
python few-shot_WangchanBERTa.py Models/WangchanBERTa4 4 > Results/few-shot_WangchanBERTa_4.out
rm -rf Models/WangchanBERTa4/Outputs
python few-shot_WangchanBERTa_MST.py Models/WangchanBERTa4 4 > Results/few-shot_WangchanBERTa_MST_4.out
rm -rf Models/WangchanBERTa4/Outputs

python fine-tune_WangchanBERTa.py Models/WangchanBERTa5 5 > Results/fine-tune_WangchanBERTa_5.out
rm -rf Models/WangchanBERTa5/Outputs
python fine-tune_WangchanBERTa_MST.py Models/WangchanBERTa5 5 > Results/fine-tune_WangchanBERTa_MST_5.out
rm -rf Models/WangchanBERTa5/Outputs
python few-shot_WangchanBERTa.py Models/WangchanBERTa5 5 > Results/few-shot_WangchanBERTa_5.out
rm -rf Models/WangchanBERTa5/Outputs
python few-shot_WangchanBERTa_MST.py Models/WangchanBERTa5 5 > Results/few-shot_WangchanBERTa_MST_5.out
rm -rf Models/WangchanBERTa5/Outputs
