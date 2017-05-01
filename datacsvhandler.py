import numpy as np
import pandas as pd
import glob
import random

'''
This script will two CSVs, one for training and one for testing of the notMNIST data set.
The idea make to it easier to use the queue functionality of tensorflow.
'''

def get_names_from_folder(filepath, filetype):
    return glob.glob(filepath+filetype)

def get_pngs_from_folder(filepath):
    return glob.glob(filepath+'*.png')

def list_of_labels():
    return ['A','B','C','D','E','F','G','H','I','J']

def img_directories():
    a = "../notMNIST/A/"
    b = "../notMNIST/B/"
    c = "../notMNIST/C/"
    d = "../notMNIST/D/"
    e = "../notMNIST/E/"
    f = "../notMNIST/F/"
    g = "../notMNIST/G/"
    h = "../notMNIST/H/"
    i = "../notMNIST/I/"
    j = "../notMNIST/J/"
    return [a,b,c,d,e,f,g,h,i,j]

def gen_repeating_list(length, value):
    return [value]*length

def gen_img_labels(directory, label):
    images = get_pngs_from_folder(directory)
    labels = gen_repeating_list(len(images), label)
    return zip(images, labels)

def gen_image_label_list():
    images = []
    directs = img_directories()
    labels = list_of_labels()
    for i in range(len(directs)):
        images = images + gen_img_labels(directs[i], labels[i])
    return images

def shuffle_list(data_pairs):
    random.shuffle(data_pairs) #shuffle is an inplace operator.

def gen_output_csv(data, filename):
    output_frame = pd.DataFrame(data)
    output_frame.to_csv(filename, index=False, header=['image filepath', 'label'])

def gen_train_test_csvs(data, split, train_name, test_name):
    idx_split = int(split*len(data))
    train_data = data[:idx_split]
    test_data = data[idx_split+1:]
    gen_output_csv(train_data, train_name)
    gen_output_csv(test_data, test_name)

imgs_and_labels = gen_image_label_list()
shuffle_list(imgs_and_labels)
# Split the data into 70% training and 30% testing:
gen_train_test_csvs(imgs_and_labels, 0.7, 'train-data.csv', 'test-data.csv')
