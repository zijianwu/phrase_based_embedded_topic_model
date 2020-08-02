import multiprocessing as mp
import os
import time

import gensim

# %%
current_path = os.path.dirname(os.path.abspath("__file__"))

training_datasets = os.listdir(f"{current_path}/../data/processed/wikipedia/")
training_datasets = [ele for ele in training_datasets if '.txt' in ele]

windows = [3, 9]
epochs_list = [10, 50]
sg_types = [0]
hs_types = [0]
size = 100


# %%

# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


def train_save_word2vec_model(dataset, min_count=2, size=size, window=3, workers=mp.cpu_count()-1, sg=1, hs=0, epochs=10):
    sentences = MySentences(f"{current_path}/../data/processed/wikipedia/{dataset}")
    # start_time = time.time()
    # print("Opening file")
    # my_file = open(f"{current_path}/../data/processed/wikipedia/{dataset}", "r")
    # sentences = my_file.read()
    # print(f"Finished reading file. Took {int(time.time() - start_time)} seconds")
    # start_time = time.time()
    # sentences = [ele.split() for ele in sentences]
    # print(f"Finished splitting sentences. Took {int(time.time() - start_time)} seconds")
    # start_time = time.time()
    # my_file.close()
    model = gensim.models.Word2Vec(sentences=sentences,
                                   min_count=min_count,
                                   size=size,
                                   window=window,
                                   workers=workers,
                                   sg=sg,  # 1 is sg, 0 is CBOW
                                   hs=hs,  # 1 is hs, 0 is ns
                                   iter=epochs
                                   )
    print(f"Finished training model. Took {int(time.time() - start_time)} seconds")
    model.save(f"{current_path}/../models/{dataset[:-4]}_word2vec_win{window}_sg{sg}_hs{hs}_epochs{epochs}.model")
    return model


def save_pretrained_embeddings(model, dataset, min_count=2, size=size, window=3, workers=mp.cpu_count()-1, sg=1, hs=0,
                               epochs=10, model_type='word2vec'):
    with open(f"{current_path}/../pretrained_embeddings/{dataset[:-4]}_{model_type}_win{window}_sg{sg}_hs{hs}_epochs{epochs}.txt", 'w') as f:
        for v in list(model.wv.vocab):
            vec = list(model.wv.__getitem__(v))
            f.write(v + ' ')
            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            f.write(vec_str + '\n')


def process_word2vec(dataset, min_count=2, size=50, window=3, workers=mp.cpu_count()-1, sg=1, hs=0, epochs=10):
    model = train_save_word2vec_model(dataset, min_count, size, window, workers, sg, hs, epochs)
    save_pretrained_embeddings(model, dataset, min_count, size, window, workers, sg, hs, epochs)


def train_save_fasttext_model(dataset, min_count=2, size=size, window=3, workers=mp.cpu_count()-1, sg=1, hs=0, epochs=10):
    sentences = MySentences(f"{current_path}/../data/processed/wikipedia/{dataset}")
    # my_file = open(f"{current_path}/../data/processed/wikipedia/{dataset}", "r")
    # sentences = my_file.read()
    # sentences = [ele.split() for ele in sentences]
    # my_file.close()
    model = gensim.models.FastText(size=size,
                                   window=window,
                                   workers = workers,
                                   min_count = min_count,
                                   sg = 1,
                                   hs = 0,
                                   iter = epochs)
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=261500, epochs=epochs)  # train
    model.save(f"{current_path}/../models/{dataset[:-4]}_fasttext_win{window}_sg{sg}_hs{hs}_epochs{epochs}.model")
    return model


def process_fasttext(dataset, min_count=2, size=size, window=3, workers=mp.cpu_count()-1, sg=1, hs=0, epochs=10):
    model = train_save_fasttext_model(dataset, min_count, size, window, workers, sg, hs, epochs)
    save_pretrained_embeddings(model, dataset, min_count, size, window, workers, sg, hs, epochs, model_type='fasttext')



# %%
for dataset in training_datasets:
    for window in windows:
        for epochs in epochs_list:
            for sg in sg_types:
                for hs in hs_types:
                    if os.path.exists(f"{current_path}/../models/{dataset[:-4]}_fasttext_win{window}_sg{sg}_hs{hs}_epochs{epochs}.model"):
                        if os.path.exists(f"{current_path}/../pretrained_embeddings/{dataset[:-4]}_fasttext_win{window}_sg{sg}_hs{hs}_epochs{epochs}.txt"):
                            continue
                        else:
                            print(f"Continuing processing of FastText dataset|{dataset} window|{window} epochs|{epochs} sg|{sg} hs|{hs}")
                            start_time = time.time()
                            model = gensim.models.FastText.load(f"{current_path}/../models/{dataset[:-4]}_fasttext_win{window}_sg{sg}_hs{hs}_epochs{epochs}.model") 
                            process_fasttext(dataset, min_count=2, size=size, window=window, workers=mp.cpu_count()-1, sg=sg,
                                         hs=hs, epochs=epochs)
                            print(f"    Processing took {int((time.time() - start_time) / 60)} min and {int((time.time() - start_time) % 60)} sec")
                    else:
                        print(f"Starting dataset|{dataset} window|{window} epochs|{epochs} sg|{sg} hs|{hs}")
                        start_time = time.time()
                        process_fasttext(dataset, min_count=2, size=size, window=window, workers=mp.cpu_count()-1, sg=sg,
                                         hs=hs, epochs=epochs)
                        print(f"    Training took {int((time.time() - start_time) / 60)} min and {int((time.time() - start_time) % 60)} sec")

# %%
for dataset in training_datasets:
    for window in windows:
        for epochs in epochs_list:
            for sg in sg_types:
                for hs in hs_types:
                    if os.path.exists(f"{current_path}/../models/{dataset[:-4]}_word2vec_win{window}_sg{sg}_hs{hs}_epochs{epochs}.model"):
                        continue
                    else:
                        print(f"Starting dataset|{dataset} window|{window} epochs|{epochs} sg|{sg} hs|{hs}")
                        start_time = time.time()
                        process_word2vec(dataset, min_count=2, size=size, window=window, workers=mp.cpu_count()-1, sg=sg,
                                         hs=hs, epochs=epochs)
                        print(f"    Training took {int((time.time() - start_time) / 60)} min and {int((time.time() - start_time) % 60)} sec")