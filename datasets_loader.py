LANGUAGE = 'english'
from torchtext import data, datasets
import stop_words
import string
import pandas as pd
import io
import numpy as np
import random

class dataset_loader():
    def __init__(self, vocab_length, vec_emb):
        self.vocab_length = vocab_length
        self.vec_emb = vec_emb
    
    def get_dataset(self, name, ):
        
        text_field, label_field = self.get_fields()
        
        if name == 'agn':
            data_path = 'data/ag_news_csv/'
            self.build_agn(data_path)
            train, test = self.get_train_test(data_path, text_field, label_field)
            
        if name == 'rt':
            data_path = 'data/movie_review/rt-polaritydata/'
            self.build_rt(data_path)
            train, test = self.get_train_test(data_path, text_field, label_field)
         
        if name == 'imdb':
            train, test = datasets.IMDB.splits(text_field, label_field)
            
        if name == 'trec-6':
            train, test = datasets.TREC.splits(text_field, label_field)
        
        if name == 'sst-1':
            train, val, test = datasets.SST.splits(text_field, label_field)
            train.examples += val.examples
        
        text_field.build_vocab(train, max_size=self.vocab_length, vectors=self.vec_emb)
        label_field.build_vocab(train)
        
        return train, test, text_field, label_field
    
#    def remove_undesired(self, x):
#        undesired = stop_words.get_stop_words(LANGUAGE) + list(string.punctuation)
#        return [ i for i in x if i not in undesired]
            
    def build_agn(self, data_path):
        train = pd.read_csv(data_path + 'train.csv', header=None)
        test = pd.read_csv(data_path + 'test.csv', header=None)

        def combine_columns_ag_news(pd):
            pd = pd.copy()
            pd[3] = pd[1] + ' ' + pd[2]
            del pd[1]
            del pd[2]
            return pd

        combine_columns_ag_news(train).to_csv(data_path + 'train_processed.csv', index=False, header=False)
        combine_columns_ag_news(test).to_csv(data_path + 'test_processed.csv', index=False, header=False)
    
    def build_rt(self, data_path):
        # Read data
        positive = []
        with io.open(data_path + "rt-polarity.pos", encoding='latin-1') as file:
            for line in file: 
                positive.append(line) #storing everything in memory!

        negative = []
        with io.open(data_path + "rt-polarity.neg", encoding='latin-1') as file:
            for line in file: 
                negative.append(line) #storing everything in memory!
                
        # Build train and test
        dataset = np.array(positive + negative)
        labels = np.array( [1]*len(positive) + [0]*len(negative) )
        train_size_ratio = 0.9
        all_data = pd.DataFrame({0:labels, 1:dataset}).sample(frac=1, random_state=8)
        all_data[:int(train_size_ratio*len(all_data))].to_csv(data_path + 'train_processed.csv', index=False, header=False)
        all_data[int(train_size_ratio*len(all_data)):].to_csv(data_path + 'test_processed.csv', index=False, header=False)
    
    def get_fields(self):
        text_field = data.Field(lower=True, include_lengths=True, batch_first=True, tokenize='spacy')#, preprocessing=self.remove_undesired)
        label_field = data.Field(sequential=False, unk_token=None)
        return text_field, label_field
    
    def get_train_test(self, data_path, text_field, label_field):
        train, test = data.TabularDataset.splits(path=data_path, 
                                                 train='train_processed.csv', 
                                                 test='test_processed.csv',
                                                 format='csv',
                                                 fields=[('label', label_field), ('text',text_field)])
        return train, test