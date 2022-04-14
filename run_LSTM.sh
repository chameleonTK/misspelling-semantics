#!/bin/bash
python train_fasttext.py Models/lstm1
python train_LSTM.py Models/cc.th.300.bin > Results/lstm1_ft.out
python train_LSTM.py Models/lstm1/fasttext_VISTEC-TP-TH-2021_misp.bin > Results/lstm1_VISTEC_misp.out
python train_LSTM.py Models/lstm1/fasttext_VISTEC-TP-TH-2021_corr.bin > Results/lstm1_VISTEC_corr.out
python train_LSTM.py Models/lstm1/fasttext_wisesight_train_misp.bin > Results/lstm1_Wisesight_misp.out

python train_LSTM_MST.py Models/lstm1/fasttext_VISTEC-TP-TH-2021_MST.bin > Results/lstm1_VISTEC_mst.out
python train_LSTM_MST.py Models/lstm1/fasttext_wisesight_train_MST.bin > Results/lstm1_Wisesight_mst.out

python train_fasttext.py Models/lstm2
python train_LSTM.py Models/cc.th.300.bin > Results/lstm2_ft.out
python train_LSTM.py Models/lstm2/fasttext_VISTEC-TP-TH-2021_misp.bin > Results/lstm2_VISTEC_misp.out
python train_LSTM.py Models/lstm2/fasttext_VISTEC-TP-TH-2021_corr.bin > Results/lstm2_VISTEC_corr.out
python train_LSTM.py Models/lstm2/fasttext_wisesight_train_misp.bin > Results/lstm2_Wisesight_misp.out

python train_LSTM_MST.py Models/lstm2/fasttext_VISTEC-TP-TH-2021_MST.bin > Results/lstm2_VISTEC_mst.out
python train_LSTM_MST.py Models/lstm2/fasttext_wisesight_train_MST.bin > Results/lstm2_Wisesight_mst.out

python train_fasttext.py Models/lstm3
python train_LSTM.py Models/cc.th.300.bin > Results/lstm3_ft.out
python train_LSTM.py Models/lstm3/fasttext_VISTEC-TP-TH-2021_misp.bin > Results/lstm3_VISTEC_misp.out
python train_LSTM.py Models/lstm3/fasttext_VISTEC-TP-TH-2021_corr.bin > Results/lstm3_VISTEC_corr.out
python train_LSTM.py Models/lstm3/fasttext_wisesight_train_misp.bin > Results/lstm3_Wisesight_misp.out

python train_LSTM_MST.py Models/lstm3/fasttext_VISTEC-TP-TH-2021_MST.bin > Results/lstm3_VISTEC_mst.out
python train_LSTM_MST.py Models/lstm3/fasttext_wisesight_train_MST.bin > Results/lstm3_Wisesight_mst.out

python train_fasttext.py Models/lstm4
python train_LSTM.py Models/cc.th.300.bin > Results/lstm4_ft.out
python train_LSTM.py Models/lstm4/fasttext_VISTEC-TP-TH-2021_misp.bin > Results/lstm4_VISTEC_misp.out
python train_LSTM.py Models/lstm4/fasttext_VISTEC-TP-TH-2021_corr.bin > Results/lstm4_VISTEC_corr.out
python train_LSTM.py Models/lstm4/fasttext_wisesight_train_misp.bin > Results/lstm4_Wisesight_misp.out

python train_LSTM_MST.py Models/lstm4/fasttext_VISTEC-TP-TH-2021_MST.bin > Results/lstm4_VISTEC_mst.out
python train_LSTM_MST.py Models/lstm4/fasttext_wisesight_train_MST.bin > Results/lstm4_Wisesight_mst.out

python train_fasttext.py Models/lstm5
python train_LSTM.py Models/cc.th.300.bin > Results/lstm5_ft.out
python train_LSTM.py Models/lstm5/fasttext_VISTEC-TP-TH-2021_misp.bin > Results/lstm5_VISTEC_misp.out
python train_LSTM.py Models/lstm5/fasttext_VISTEC-TP-TH-2021_corr.bin > Results/lstm5_VISTEC_corr.out
python train_LSTM.py Models/lstm5/fasttext_wisesight_train_misp.bin > Results/lstm5_Wisesight_misp.out

python train_LSTM_MST.py Models/lstm5/fasttext_VISTEC-TP-TH-2021_MST.bin > Results/lstm5_VISTEC_mst.out
python train_LSTM_MST.py Models/lstm5/fasttext_wisesight_train_MST.bin > Results/lstm5_Wisesight_mst.out