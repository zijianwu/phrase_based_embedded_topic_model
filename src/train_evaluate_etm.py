import os
import time

current_path = os.path.dirname(os.path.abspath("__file__"))

targets = os.listdir(f"{current_path}/../data/modeling/20_newsgroups")
targets = [ele for ele in targets if 'DS_Store' not in ele]

pretrained_embeddings = os.listdir(f"{current_path}/../pretrained_embeddings")
pretrained_embeddings = [ele for ele in pretrained_embeddings if '.txt' in ele]

#%%

for target in sorted(targets):
    for pretrained_embedding in sorted(pretrained_embeddings):
        if os.path.exists(f"{current_path}/../results/{target}_{pretrained_embedding[:-4]}_test_output.txt"):
            print(f"{target}_{pretrained_embedding[:-4]} exists")
            continue
        else:
            print(f"Starting {target}_{pretrained_embedding[:-4]}")
            start_time = time.time()
            training_script = f"python main.py --mode train --dataset 20ng --data_path ../data/modeling/20_newsgroups/{target} --emb_path ../pretrained_embeddings/{pretrained_embedding} --num_topics 20 --train_embeddings 0 --epochs 100 --rho_size 100 --emb_size 100 --save_path ../outputs/{target}_{pretrained_embedding[:-4]}/ > ../results/{target}_{pretrained_embedding[:-4]}_training_output.txt"
            os.system(training_script)
            print(f"Took {int(time.time() - start_time)} to train")
            start_time = time.time()
            testing_script = f"python main.py --mode eval --dataset 20ng --data_path ../data/modeling/20_newsgroups/{target} --emb_path ../pretrained_embeddings/{pretrained_embedding} --num_topics 20 --train_embeddings 0 --epochs 100 --rho_size 100 --emb_size 100 --load_from ../outputs/{target}_{pretrained_embedding[:-4]}/etm_20ng_K_20_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_100_trainEmbeddings_0 --tc 1 --td 1 > ../results/{target}_{pretrained_embedding[:-4]}_test_output.txt"
            os.system(testing_script)
            print(f"Took {int(time.time() - start_time)} to test")

