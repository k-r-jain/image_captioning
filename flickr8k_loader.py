import errno
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import warnings
from os.path import dirname
from skimage import io, transform

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


warnings.filterwarnings("ignore")

class Flickr8kDataset(Dataset):

    def __init__(self, root_dir, mode = 'train', transform = None, vocab_path = 'models/vocab.pkl'):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.vocab_path = vocab_path

        self.wordtoidx = {}
        self.idxtoword = {}
        self.vocab_idx = 0
        self.max_sequence_length = 0
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.pad_token = '<pad>'

        self.captions_dir = 'captions'
        self.images_dir = 'images'
        self.captions = pd.read_csv(os.path.join(self.root_dir, self.captions_dir, 'Flickr8k.token.txt'), sep = '\t', header = None, names = ['image', 'caption'])
        self.captions['image'], self.captions['caption_number'] = self.captions['image'].str.split('#', 1).str
        self.train_set = pd.read_csv(os.path.join(self.root_dir, self.captions_dir, 'Flickr_8k.trainImages.txt'), header = None, names = ['image'])
        self.val_set = pd.read_csv(os.path.join(self.root_dir, self.captions_dir, 'Flickr_8k.devImages.txt'), header = None, names = ['image'])
        self.test_set = pd.read_csv(os.path.join(self.root_dir, self.captions_dir, 'Flickr_8k.testImages.txt'), header = None, names = ['image'])

        path = os.path.join(os.getcwd(), self.vocab_path)
        print(path)
        if os.path.exists(path):
            print('Found vocab file at', self.vocab_path)
            [self.wordtoidx, self.idxtoword, self.max_sequence_length, self.vocab_idx] = pickle.load(open(path, 'rb'))
            print(len(self.wordtoidx), len(self.idxtoword))
        else:
            print('Did not find a vocab file. Creating one ...')
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            self.build_vocab()
            pickle.dump([self.wordtoidx, self.idxtoword, self.max_sequence_length, self.vocab_idx], open(path, 'wb'))
            print('Saved vocab at:', path)
            print(len(self.wordtoidx), len(self.idxtoword))
            



    def add_to_vocab(self, word):
        if not word in self.wordtoidx:
            self.wordtoidx[word] = self.vocab_idx
            self.idxtoword[self.vocab_idx] = word
            self.vocab_idx += 1

    def build_vocab(self):
        self.add_to_vocab(self.pad_token)
        self.add_to_vocab(self.start_token)
        self.add_to_vocab(self.end_token)
        for caption in self.captions['caption'].as_matrix():
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            if len(tokens) > self.max_sequence_length:
                self.max_sequence_length = len(tokens)
            # print(tokens)
            for word in tokens:
                self.add_to_vocab(word)

    def get_vocab_size(self):
        # print(self.vocab_idx)
        return len(self.wordtoidx)

    def __len__(self):
        if self.mode == 'val':
            return len(self.val_set)
        elif self.mode == 'test':
            return len(self.test_set)
        else:
            return len(self.train_set)
    
    def __getitem__(self, idx):
        if self.mode == 'val':
            image_name = self.val_set.iloc[idx]['image']
        elif self.mode == 'test':
            image_name = self.test_set.iloc[idx]['image']
        else:
            image_name = self.train_set.iloc[idx]['image']
        
        image = io.imread(os.path.join(self.root_dir, self.images_dir, image_name))
        if self.transform is not None:
            image = self.transform(image)
        
        image_captions = self.captions.loc[self.captions['image'] == image_name]['caption'].as_matrix()
        # print(image_name, image_captions)

        tokenized_captions = []
        for caption in image_captions:
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            tokens.insert(0, self.start_token)
            tokens.append(self.end_token)
            while len(tokens) < (self.max_sequence_length + 2): # 2 for start and end tags
                tokens.append(self.pad_token)
            tokenized_captions.append([self.wordtoidx[word] for word in tokens])

        return {'image': image, 'name': image_name, 'captions': torch.Tensor(np.array(tokenized_captions, dtype = 'int'))}
