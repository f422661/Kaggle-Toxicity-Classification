import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from keras.preprocessing import text, sequence


class TextDataset(Dataset):

    def __init__(self,x,y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx],self.y[idx]
        return self.x[idx]


def abc():
	a = 2
	b = 3
	return a,b

def collate_fn_train(batch):

  feature = [x[0] for x in batch]
  label = [x[1] for x in batch]

  feature = sequence.pad_sequences(feature,padding='post')
  feature = torch.tensor(feature, dtype=torch.long)
  label = torch.tensor(label, dtype=torch.float)
  
  return feature, label

def collate_fn_test(batch):

  feature = sequence.pad_sequences(batch,padding='post')
  feature = torch.tensor(feature, dtype=torch.long)
  
  return feature


tokenizer = text.Tokenizer()
x = [['1','23','23','1','2','5'],['1','23','23','12','5'],['2','23','2'],['1']]

tokenizer.fit_on_texts(x)

# x_train = tokenizer.texts_to_sequences(x_train)
# x_test = tokenizer.texts_to_sequences(x_test)

# x = [[1,23,23,1,2,5],[1,23,23,12,5],[2,23,2],[1]]

# x = [[1,1],[1,1],[2,2],[2,2]]



y = [1,1,1,0] 


train_data = TextDataset(x)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True,collate_fn =collate_fn_test)

result = abc()
for data in train_loader:
	# inputs, labels = data

	print(data)
	# print(labels)
    # print(data)
