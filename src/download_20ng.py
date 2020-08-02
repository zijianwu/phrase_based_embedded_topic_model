from sklearn.datasets import fetch_20newsgroups
import pickle
import os

current_path = os.path.dirname(os.path.abspath("__file__"))

newsgroups_train_data_loc = f"{current_path}/../data/raw/20_newsgroups/train_data.pkl"
newsgroups_train_labels_loc = f"{current_path}/../data/raw/20_newsgroups/train_labels.pkl"
newsgroups_test_data_loc = f"{current_path}/../data/raw/20_newsgroups/test_data.pkl"
newsgroups_test_labels_loc = f"{current_path}/../data/raw/20_newsgroups/test_labels.pkl"

os.makedirs(os.path.dirname(newsgroups_train_data_loc), exist_ok=True)
os.makedirs(os.path.dirname(newsgroups_train_labels_loc), exist_ok=True)
os.makedirs(os.path.dirname(newsgroups_test_data_loc), exist_ok=True)
os.makedirs(os.path.dirname(newsgroups_test_labels_loc), exist_ok=True)

train_data, train_labels = fetch_20newsgroups(subset='train', return_X_y=True)
test_data, test_labels = fetch_20newsgroups(subset='test', return_X_y=True)

pickle.dump(train_data, open(newsgroups_train_data_loc, "wb"))
pickle.dump(train_labels, open(newsgroups_train_labels_loc, "wb"))
pickle.dump(test_data, open(newsgroups_test_data_loc, "wb"))
pickle.dump(test_labels, open(newsgroups_test_labels_loc, "wb"))