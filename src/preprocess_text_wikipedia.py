import pickle
import os
import spacy

import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from spacy.util import minibatch
from spacy.attrs import LEMMA
from multiprocessing import Process
import time
import itertools 

nlp = spacy.load("en_core_web_sm", disable=[ #"tagger",
    # "parser",
    # "ner"
])

import gensim.downloader as api

#%%

current_path = os.path.dirname(os.path.abspath("__file__"))

#%%

def parallel_apply_list(a_list, a_function, n_jobs=mp.cpu_count(), func_param=None, n_threads=None, **kwargs):
    """
    Applies a_function to a_list using multiprocessing with n_jobs. If a_function has a specific
    parameter that elements in a_list should fill, indicate it with func_param. If there are
    other parameters in a_function that should be statically filled, use **kwargs.

    If elements in a_list are tuples, lists, or anything else that provides multiple inputs
    to a_function, wrap a_function so that it takes the entire tuple or list (see parallel_apply_row for example)

    Parameters
    ----------
    a_list : list
    a_function : function object
    n_jobs : int (multiprocessing)
    n_threads : int (threading)
    func_param : None (defaults to first parameter in function) or str (parameter in a_function)
    kwargs : static keyword arguments to be given to all instances of a_function

    Returns
    -------
    result : list
    """
    if n_jobs:
        executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    else:
        executor = Parallel(n_jobs=n_threads, backend="threading", prefer="threads")
    do = delayed(partial(a_function, **kwargs))
    if func_param:
        tasks = (do(**{func_param: ele}) for ele in tqdm(a_list))
    else:
        tasks = (do(ele) for ele in tqdm(a_list))
    result = executor(tasks)
    return result

def spacy_norm(text):
    doc = nlp(text)
    tokenized_doc = [ele.text.lower() for ele in doc if (not ele.is_space) and (not ele.is_punct)]
    result = ' '.join(tokenized_doc) + '\n'
    return result
def process_save_norm(texts, data_name, data_type, current_path):
    norm_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_norm.txt"
    if os.path.exists(norm_outpath):
        norm_outpath_write = 'a'
    else:
        norm_outpath_write = 'w'
        
    os.makedirs(os.path.dirname(norm_outpath), exist_ok=True)
    results = parallel_apply_list(texts, spacy_norm)
    f0 = open(norm_outpath, norm_outpath_write)
    f0.writelines(results)
    f0.close()

def spacy_lemma(text):
    doc = nlp(text)
    results = [ele.lemma_.lower() for ele in doc if (not ele.is_space) and (not ele.is_punct)]
    results = ' '.join(results) + '\n'
    return results
def process_save_lemma(texts, data_name, data_type, current_path):
    lemma_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_lemma.txt"
    if os.path.exists(lemma_outpath):
        lemma_outpath_write = 'a'
    else:
        lemma_outpath_write = 'w'
    
    os.makedirs(os.path.dirname(lemma_outpath), exist_ok=True)
    results = parallel_apply_list(texts, spacy_lemma)
    f1 = open(lemma_outpath, lemma_outpath_write)
    f1.writelines(results)
    f1.close()       

def spacy_no_stop(text):
    doc = nlp(text)
    results = [ele.text.lower() for ele in doc if ((not ele.is_space) and (not ele.is_punct) and (not ele.is_stop))]
    results = ' '.join(results) + '\n'
    return results
def process_save_no_stop(texts, data_name, data_type, current_path):
    no_stop_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_no_stop.txt"
    if os.path.exists(no_stop_outpath):
        no_stop_outpath_write = 'a'
    else:
        no_stop_outpath_write = 'w'
        
    os.makedirs(os.path.dirname(no_stop_outpath), exist_ok=True)
    results = parallel_apply_list(texts, spacy_no_stop)
    f2 = open(no_stop_outpath, no_stop_outpath_write)
    f2.writelines(results)
    f2.close()
    
def spacy_lemma_no_stop(text):
    doc = nlp(text)
    results = [ele.lemma_.lower() for ele in doc if ((not ele.is_space) and (not ele.is_punct) and (not ele.is_stop))]
    results = ' '.join(results) + '\n'
    return results
def process_save_lemma_no_stop(texts, data_name, data_type, current_path):
    lemma_no_stop_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_lemma_no_stop.txt"
    if os.path.exists(lemma_no_stop_outpath):
        lemma_no_stop_outpath_write = 'a'
    else:
        lemma_no_stop_outpath_write = 'w'
    
    os.makedirs(os.path.dirname(lemma_no_stop_outpath), exist_ok=True)
    results = parallel_apply_list(texts, spacy_lemma_no_stop)
    f3 = open(lemma_no_stop_outpath, lemma_no_stop_outpath_write)
    f3.writelines(results)
    f3.close()

def spacy_np_lemma(text):
    doc = nlp(text.replace('\n', ' '))
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
            np = np[1:]
        if len(np) > 1:
            with doc.retokenize() as retokenizer:
                doc.vocab['_'.join([ele.lemma_.lower() for ele in np])]
                retokenizer.merge(np, attrs={LEMMA: doc.vocab.strings['_'.join([ele.lemma_.lower() for ele in np])]})
    for ent in doc.ents:
        if len(ent) > 1:
            with doc.retokenize() as retokenizer:
                doc.vocab['_'.join([ele.lemma_.lower() for ele in ent])]
                retokenizer.merge(ent, attrs={LEMMA: doc.vocab.strings['_'.join([ele.lemma_.lower() for ele in ent])]})

    tokenized_doc = [ele.lemma_.lower().replace(' ', '_') for ele in doc if ((not ele.is_space) and (not ele.is_punct))]
    tokenized_doc = [ele for ele in tokenized_doc if ele]
    return tokenized_doc
def extract_noun_phrases_from_tok_list(a_list_of_toks):
    results = [ele for ele in a_list_of_toks if '_' in ele]
    return results
def join_list_of_strings_add_newline(a_list_of_toks):
    result = ' '.join(a_list_of_toks) + '\n'
    return result
def process_save_np_lemma(texts, data_name, data_type, current_path):
    np_lemma_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_np_lemma.txt"
    if os.path.exists(np_lemma_outpath):
        np_lemma_outpath_write = 'a'
    else:
        np_lemma_outpath_write = 'w'
    
    os.makedirs(os.path.dirname(np_lemma_outpath), exist_ok=True)
    results = parallel_apply_list(texts, spacy_np_lemma)
    result1 = parallel_apply_list(results, join_list_of_strings_add_newline) 
    f5 = open(np_lemma_outpath, np_lemma_outpath_write)
    f5.writelines(result1)
    f5.close()
    
    np_lemma_outpath_only = f"{current_path}/../data/processed/{data_name}/{data_type}_np_lemma_only.txt"
    if os.path.exists(np_lemma_outpath_only):
        np_lemma_outpath_write_only = 'a'
    else:
        np_lemma_outpath_write_only = 'w'
    
    os.makedirs(os.path.dirname(np_lemma_outpath_only), exist_ok=True)
    result2 = parallel_apply_list(results, extract_noun_phrases_from_tok_list)
    result2 = [' '.join(ele) + '\n' for ele in result2]
    f8 = open(np_lemma_outpath_only, np_lemma_outpath_write_only)
    f8.writelines(result2)
    f8.close()
    
def spacy_np_no_stop(text):
    doc = nlp(text.replace('\n', ' '))
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
            np = np[1:]
        if len(np) > 1:
            with doc.retokenize() as retokenizer:
                doc.vocab['_'.join([ele.text.lower() for ele in np])]
                retokenizer.merge(np, attrs={LEMMA: doc.vocab.strings['_'.join([ele.text.lower() for ele in np])]})
    for ent in doc.ents:
        if len(ent) > 1:
            with doc.retokenize() as retokenizer:
                doc.vocab['_'.join([ele.text.lower() for ele in ent])]
                retokenizer.merge(ent, attrs={LEMMA: doc.vocab.strings['_'.join([ele.text.lower() for ele in ent])]})

    tokenized_doc = [ele.text.lower().strip().replace(' ', '_') for ele in doc if ((not ele.is_stop) and (not ele.is_space) and (not ele.is_punct))]
    tokenized_doc = [ele for ele in tokenized_doc if ele]
    return tokenized_doc
def process_save_np_no_stop(texts, data_name, data_type, current_path):
    np_no_stop_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_np_no_stop.txt"
    if os.path.exists(np_no_stop_outpath):
        np_no_stop_outpath_write = 'a'
    else:
        np_no_stop_outpath_write = 'w'
        
    os.makedirs(os.path.dirname(np_no_stop_outpath), exist_ok=True)
    start_time = time.time()
    results = parallel_apply_list(texts, spacy_np_no_stop)
    print(f"        Finished spacy_np_no_stop, taking {int(time.time() - start_time)} seconds...")
    start_time = time.time()
    result1 = parallel_apply_list(results, join_list_of_strings_add_newline) 
    print(f"        Finished join_list_of_strings_add_newline, taking {int(time.time() - start_time)} seconds...")
    start_time = time.time()
    f4 = open(np_no_stop_outpath, np_no_stop_outpath_write)
    f4.writelines(result1)
    f4.close()
    print(f"        Finished writing to file, taking {int(time.time() - start_time)} seconds...")
    
    np_no_stop_outpath_only = f"{current_path}/../data/processed/{data_name}/{data_type}_np_no_stop_only.txt"
    if os.path.exists(np_no_stop_outpath_only):
        np_no_stop_outpath_write_only = 'a'
    else:
        np_no_stop_outpath_write_only = 'w'
        
    os.makedirs(os.path.dirname(np_no_stop_outpath_only), exist_ok=True)
    result2 = parallel_apply_list(results, extract_noun_phrases_from_tok_list)
    result2 = [' '.join(ele) + '\n' for ele in result2]
    f7 = open(np_no_stop_outpath_only, np_no_stop_outpath_write_only)
    f7.writelines(result2)
    f7.close()
     
def spacy_np_lemma_no_stop(text):
    doc = nlp(text.replace('\n', ' '))
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
            np = np[1:]
        if len(np) > 1:
            with doc.retokenize() as retokenizer:
                doc.vocab['_'.join([ele.lemma_.lower() for ele in np])]
                retokenizer.merge(np, attrs={LEMMA: doc.vocab.strings['_'.join([ele.lemma_.lower() for ele in np])]})
    for ent in doc.ents:
        if len(ent) > 1:
            with doc.retokenize() as retokenizer:
                doc.vocab['_'.join([ele.lemma_.lower() for ele in ent])]
                retokenizer.merge(ent, attrs={LEMMA: doc.vocab.strings['_'.join([ele.lemma_.lower() for ele in ent])]})

    tokenized_doc = [ele.lemma_.lower().replace(' ', '_') for ele in doc if ((not ele.is_stop) and (not ele.is_space) and (not ele.is_punct))]
    tokenized_doc = [ele for ele in tokenized_doc if ele]
    tokenized_doc = ' '.join(tokenized_doc) + '\n'
    return tokenized_doc
def process_save_np_lemma_no_stop(texts, data_name, data_type, current_path):
    np_lemma_no_stop_outpath = f"{current_path}/../data/processed/{data_name}/{data_type}_np_lemma_no_stop.txt"
    if os.path.exists(np_lemma_no_stop_outpath):
        np_lemma_no_stop_outpath_write = 'a'
    else:
        np_lemma_no_stop_outpath_write = 'w'
    
    os.makedirs(os.path.dirname(np_lemma_no_stop_outpath), exist_ok=True)
    results = parallel_apply_list(texts, spacy_np_lemma_no_stop)  
    f6 = open(np_lemma_no_stop_outpath, np_lemma_no_stop_outpath_write)           
    f6.writelines(results)
    f6.close()
    
    
def fully_process_partitions(partition, data_name='20_newgroups', data_type='train', current_path=current_path):
    # print("    Processing norm...")
    # process_save_norm(texts=partition,
    #                   data_name=data_name,
    #                   data_type=data_type,
    #                   current_path=current_path)
    # print("    Processing lemma...")
    # process_save_lemma(texts=partition,
    #                    data_name=data_name,
    #                    data_type=data_type,
    #                    current_path=current_path)
    # print("    Processing no_stop...")
    # process_save_no_stop(texts=partition,
    #                    data_name=data_name,
    #                    data_type=data_type,
    #                    current_path=current_path)
    # print("    Processing lemma_no_stop...")
    # process_save_lemma_no_stop(texts=partition,
    #                    data_name=data_name,
    #                    data_type=data_type,
    #                    current_path=current_path)
    print("    Processing np_lemma...")
    process_save_np_lemma(texts=partition,
                       data_name=data_name,
                       data_type=data_type,
                       current_path=current_path)
    print("    Processing np_no_stop...")
    process_save_np_no_stop(texts=partition,
                       data_name=data_name,
                       data_type=data_type,
                       current_path=current_path)
    print("    Processing np_lemma_no_stop...")
    process_save_np_lemma_no_stop(texts=partition,
                       data_name=data_name,
                       data_type=data_type,
                       current_path=current_path)
    
    
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        chunk = [ele['section_texts'] for ele in chunk]
        chunk = [ele for sublist in chunk for ele in sublist]
        chunk = [ele if len(ele) > 999998 else ele[:999998] for ele in chunk]
        yield chunk


#%%

corpus = api.load('wiki-english-20171001')
partitions = grouper(1000, corpus)

#%%
iter_number = 0
while iter_number < 30:
    iter_number += 1
    partition = next(partitions)
    print(f"Starting partition {iter_number}")
    fully_process_partitions(partition, data_name='wikipedia', data_type='train')
