import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datetime import datetime

import pandas as pd
import json
import math
import random

from sklearn.model_selection import train_test_split

SOS = -1
EOS = 0

def parse_time(x):
  # We are using python's builtin datetime library
  # https://docs.python.org/3/library/datetime.html#datetime.date.fromtimestamp

  # Each x is essentially a 1 row, 1 column pandas Series
  dt = datetime.fromtimestamp(x["TIMESTAMP"])
  return dt.year, dt.month, dt.day, dt.hour, dt.weekday()

def map_one_hot(val):
    if (val == 'A'):
        return 0
    elif (val == 'B'):
        return 1
    elif (val == 'C'):
        return 2
    
def map_year(val):
    if val < 2014:
        return 1
    else:
        return 0
    
def map_json(val):
    return json.loads(val)

def map_check_na(val):
    return 0 if math.isnan(val) else val

def map_normalize(val):
    normal = []
    for v in val:
        normal.append((v - 4502.893653516295) / 215.0430544729272)
        
    return normal

def map_red(val):
    return val - 1

# explained by https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
# Input is list of tuples of the form (enc_feats, coords, travel_time) with len batch_size
# Output should be two tuples
#     - [0]: Encoder input -> tensor of shape [batch_size, 594] (should become [batch, 594]
#     - [1]: Decoder input -> tensor of shape [seq_len, batch_size, 1] (explained further in notes)
#     - [2]: Travel times
def collate_fn(data):
    coord_list = []
    feat_list = []
    time_list = []
    for i in range(len(data)):
        coord_list.append(data[i][1])
        feat_list.append(data[i][0])
        time_list.append(data[i][2])
        
    coord_list = tensorFromCoords(coord_list)
    coord_list = coord_list[0].unsqueeze(-1) # feature for decoder
    feat_list = torch.stack(feat_list, 0)
    time_list = torch.stack(time_list, 0)

    return (feat_list, coord_list, time_list)

def collate_fn_test(data):
    feat_list = []
    tid_list = []
    for i in range(len(data)):
        tid_list.append(data[i][0])
        feat_list.append(data[i][1])
        
    feat_list = torch.stack(feat_list, 0)

    return (tid_list, feat_list)
    
""" Helper Functions for Batches """   
def tensorFromCoords(coord_list):
    # Takes in a batch of list of coords

    d = [torch.Tensor([SOS] + i + [EOS]) for i in coord_list]
    c = nn.utils.rnn.pack_sequence(d, enforce_sorted=False)
    c = nn.utils.rnn.pad_packed_sequence(c)
    
    # Returns a tuple, (padded coord, batch)
    return c


class TaxiDataset(Dataset):
    def __init__(self, df):        
        # Convert df feats into tensor compatible formats and one hot encode
        df[["YR", "MON", "DAY", "HR", "WK"]] = df[["TIMESTAMP"]].apply(parse_time, axis=1, result_type="expand")
        
        self.taxi_id = F.one_hot(torch.Tensor(df['TAXI_ID'].values).to(torch.int64), 448)
        self.day_type = F.one_hot(torch.Tensor(df['DAY_TYPE'].map(map_one_hot).values).to(torch.int64), 3)
        self.call_type = F.one_hot(torch.Tensor(df['CALL_TYPE'].map(map_one_hot).values).to(torch.int64), 3)
        self.origin_stand = F.one_hot(torch.Tensor(df['ORIGIN_STAND'].map(map_check_na).values).to(torch.int64), 64)
        self.yr = F.one_hot(torch.Tensor(df['YR'].map(map_year).values).to(torch.int64), 2)
        self.mon = F.one_hot(torch.Tensor(df['MON'].map(map_red).values).to(torch.int64), 12)
        self.day = F.one_hot(torch.Tensor(df['DAY'].map(map_red).values).to(torch.int64), 31)
        self.hr = F.one_hot(torch.Tensor(df['HR'].values).to(torch.int64), 24)
        self.wk = F.one_hot(torch.Tensor(df['WK'].values).to(torch.int64), 7)
        self.travel_time = torch.Tensor(df['TRAVEL_TIME'].values)
        self.coords = df['COORDS'].map(map_json).tolist()
        
        
    def __len__(self):
        return len(self.taxi_id)
        
    def __getitem__(self, idx):
        enc_feats = torch.cat((self.taxi_id[idx], self.day_type[idx],self.call_type[idx],self.origin_stand[idx],
                               self.yr[idx],self.mon[idx],self.day[idx],self.hr[idx],self.wk[idx]))
        coords = self.coords[idx]
        travel_time = self.travel_time[idx]
        
        return enc_feats, coords, travel_time
    
class PredDataset(Dataset):
    def __init__(self, df):
        # Convert df feats into tensor compatible formats and one hot encode
        df[["YR", "MON", "DAY", "HR", "WK"]] = df[["TIMESTAMP"]].apply(parse_time, axis=1, result_type="expand")
        
        self.tid = df['TRIP_ID']
        self.taxi_id = F.one_hot(torch.Tensor(df['TAXI_ID'].values).to(torch.int64), 448)
        self.day_type = F.one_hot(torch.Tensor(df['DAY_TYPE'].map(map_one_hot).values).to(torch.int64), 3)
        self.call_type = F.one_hot(torch.Tensor(df['CALL_TYPE'].map(map_one_hot).values).to(torch.int64), 3)
        self.origin_stand = F.one_hot(torch.Tensor(df['ORIGIN_STAND'].map(map_check_na).values).to(torch.int64), 64)
        self.yr = F.one_hot(torch.Tensor(df['YR'].map(map_year).values).to(torch.int64), 2)
        self.mon = F.one_hot(torch.Tensor(df['MON'].map(map_red).values).to(torch.int64), 12)
        self.day = F.one_hot(torch.Tensor(df['DAY'].map(map_red).values).to(torch.int64), 31)
        self.hr = F.one_hot(torch.Tensor(df['HR'].values).to(torch.int64), 24)
        self.wk = F.one_hot(torch.Tensor(df['WK'].values).to(torch.int64), 7)
        
    def __len__(self):
        return len(self.taxi_id)
    
    def __getitem__(self,idx):
        return self.tid[idx], torch.cat((self.taxi_id[idx], self.day_type[idx],self.call_type[idx],self.origin_stand[idx],
                          self.yr[idx],self.mon[idx],self.day[idx],self.hr[idx],self.wk[idx]))


    

def map_id(val, t_id):
    return t_id[val]


def get_loader(batch_size=128, test = False):
    if test:
        df = pd.read_csv('data/test_public.csv')
        df = df[df['MISSING_DATA'] == False]
        
        t_id = {}
        id_ctr = 0

        for i in range(len(df)):
            taxi_id = df['TAXI_ID'].iloc[i]
            try:
                t_id[taxi_id]
            except:
                t_id[taxi_id] = id_ctr
                id_ctr += 1
                
        df['TAXI_ID'] = df['TAXI_ID'].apply(lambda x: map_id(x, t_id))
        
        test_set = PredDataset(df)
        test_loader = DataLoader(test_set, 1, shuffle=False, collate_fn=collate_fn_test)
        
        return test_loader, len(test_set)

        
    df = pd.read_csv('data/train_up.csv')
    df = df.drop(['TRIP_ID'], axis=1)
    df = df[df['TRAVEL_TIME'] > 50]

    # Intialize a training dataset and a validation dataset from subsets of the overall dataset
    train_split, test_split = train_test_split(df, shuffle = True)

    del(df) # Free up memory

    
    train_set = TaxiDataset(train_split)
    test_set = TaxiDataset(test_split)


    train_dl = DataLoader(train_set, batch_size = 64, shuffle = True, collate_fn = collate_fn)
    test_dl = DataLoader(test_set, batch_size = 64, shuffle = True, collate_fn = collate_fn)
    
    return train_dl, test_dl, len(train_set), len(test_set)