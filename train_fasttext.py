import fasttext
import sys
# unsupervised_default = {
#         'model': "skipgram",
#       'lr': 0.05,
#     'dim': 100,
#     'ws': 5,
#     'epoch': 5,
#     'minCount': 5,
#     'minCountLabel': 0,
#     'minn': 3,
#     'maxn': 6,
#     'neg': 5,
#     'wordNgrams': 1,
#     'loss': "ns",
#     'bucket': 2000000,
#     'thread': multiprocessing.cpu_count() - 1,
#     'lrUpdateRate': 100,
#     't': 1e-4,
#     'label': "__label__",
#     'verbose': 2,
#     'pretrainedVectors': "",
#     'seed': 0,
#     'autotuneValidationFile': "",
#     'autotuneMetric': "f1",
#     'autotunePredictions': 1,
#     'autotuneDuration': 60 * 5,  # 5 minutes
#     'autotuneModelSize': ""
# }


if __name__ == '__main__':

    DIR = "Datasets"
    if len(sys.argv) < 2:
        raise Exception("Please specify output_dir")

    outputDIR = sys.argv[1]

    trainingDataFile = f"{DIR}/VISTEC-TP-TH-2021_fasttext_training_misp.txt"
    model = fasttext.train_unsupervised(trainingDataFile, "cbow", neg=10)
    for vec in model.get_nearest_neighbors("แมว"):
        print(vec)
    model.save_model(f"{outputDIR}/fasttext_VISTEC-TP-TH-2021_misp.bin")


    trainingDataFile = f"{DIR}/VISTEC-TP-TH-2021_fasttext_training_corr.txt"
    model = fasttext.train_unsupervised(trainingDataFile, "cbow", neg=10)
    for vec in model.get_nearest_neighbors("แมว"):
        print(vec)
    model.save_model(f"{outputDIR}/fasttext_VISTEC-TP-TH-2021_corr.bin")


    trainingDataFile = f"{DIR}/VISTEC-TP-TH-2021_fasttext_training_MST.txt"
    model = fasttext.train_unsupervised(trainingDataFile, "cbow", neg=10)
    for vec in model.get_nearest_neighbors("แมว"):
        print(vec)
    model.save_model(f"{outputDIR}/fasttext_VISTEC-TP-TH-2021_MST.bin")




    trainingDataFile = f"{DIR}/wisesight_train_fasttext_training_misp.txt"
    model = fasttext.train_unsupervised(trainingDataFile, "cbow", neg=10)
    for vec in model.get_nearest_neighbors("แมว"):
        print(vec)
    model.save_model(f"{outputDIR}/fasttext_wisesight_train_misp.bin")


    trainingDataFile = f"{DIR}/wisesight_train_fasttext_training_MST.txt"
    model = fasttext.train_unsupervised(trainingDataFile, "cbow", neg=10)
    for vec in model.get_nearest_neighbors("แมว"):
        print(vec)
    model.save_model(f"{outputDIR}/fasttext_wisesight_train_MST.bin")





