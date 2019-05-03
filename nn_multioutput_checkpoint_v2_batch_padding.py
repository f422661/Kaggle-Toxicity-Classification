import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import re
import torch.nn.functional as F
from keras.preprocessing import text, sequence
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm

CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
# GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'

BATCH_SIZE = 512
EPOCHS = 4
MAX_LEN = 220

misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}

def _get_misspell(misspell_dict):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re


def replace_typical_misspell(text):
    misspellings, misspellings_re = _get_misspell(misspell_dict)

    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)
    

def preprocess(data):

  '''
  Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
  '''

  punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
  def clean_special_chars(text, punct):
    for p in punct:
      text = text.replace(p, ' ')
    return text

  ## lower
  # data = data.str.lower()

  ## replace typical misspell
  data = data.astype(str).apply(replace_typical_misspell)

  ## clean the text
  data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

  return data

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def collate_fn_train(batch):

  feature = [x[0] for x in batch]
  label = [x[1] for x in batch]

  feature = sequence.pad_sequences(feature,padding='post')
  feature = torch.tensor(feature, dtype=torch.long).cuda()
  label = torch.tensor(label, dtype=torch.float).cuda()
  
  return feature, label

def collate_fn_test(batch):

  feature = sequence.pad_sequences(batch,padding='post')
  feature = torch.tensor(feature, dtype=torch.long).cuda()
  
  return feature


def build_matrix(word_index, path):

	"""
	Credit goes to https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
	"""

	embedding_index = load_embeddings(path)
	embedding_matrix = np.zeros((len(word_index) + 1, 300))
	unknown_words = []

	for word, i in word_index.items():
			try:
					embedding_matrix[i] = embedding_index[word]
			except KeyError:
					unknown_words.append(word)
	return embedding_matrix, unknown_words


def custom_loss(data, targets):

    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss = nn.BCELoss(weight=targets[:,1])(data[:,0],targets[:,0])
    return bce_loss

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


class Attention(nn.Module):
	def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
		super(Attention, self).__init__(**kwargs)
		
		self.supports_masking = True
		self.bias = bias
		self.feature_dim = feature_dim
		self.step_dim = step_dim
		self.features_dim = 0
		
		weight = torch.zeros(feature_dim, 1)
		nn.init.xavier_uniform_(weight)
		self.weight = nn.Parameter(weight)
		
		if bias:
				self.b = nn.Parameter(torch.zeros(1))
        
	def forward(self, x, mask=None):

		feature_dim = self.feature_dim
		step_dim = self.step_dim

		eij = torch.mm(
				x.contiguous().view(-1, feature_dim), 
				self.weight
		).view(-1, step_dim)
		
		if self.bias:
				eij = eij + self.b
				
		eij = torch.tanh(eij)
		a = torch.exp(eij)
		
		if mask is not None:
				a = a * mask

		a = a / torch.sum(a, 1, keepdim=True) + 1e-10

		weighted_input = x * torch.unsqueeze(a, -1)
		return torch.sum(weighted_input, 1)

class SpatialDropout(nn.Module):

	def __init__(self,p):
		super(SpatialDropout, self).__init__()
		self.dropout = nn.Dropout2d(p)

	def forward(self, x):

			x = x.permute(0, 2, 1)   # convert to [batch, feature, timestep]
			x = self.dropout(x)
			x = x.permute(0, 2, 1)   # back to [batch, timestep, feature]
			return x

class NeuralNet(nn.Module):

	def __init__(self,embedding_matrix,num_unit):
		super(NeuralNet, self).__init__()
		self.max_feature = embedding_matrix.shape[0]
		self.embedding_size = embedding_matrix.shape[1]
		self.embedding = nn.Embedding(self.max_feature, self.embedding_size)
		self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
		self.embedding.weight.requires_grad = False
		self.embedding_dropout = SpatialDropout(0.3)
		self.lstm1 = nn.LSTM(self.embedding_size, num_unit, bidirectional=True, batch_first=True)
		self.lstm2 = nn.LSTM(num_unit*2, int(num_unit/2), bidirectional=True, batch_first=True)
		# self.attention = Attention(num_unit, MAX_LEN)
		self.linear1 = nn.Linear(num_unit*2, num_unit)
		self.linear2 = nn.Linear(num_unit*2, num_unit)
		self.linear_out = nn.Linear(num_unit, 1)
		self.linear_aux_out = nn.Linear(num_unit, 5)

	def forward(self, x):

		h_embedding = self.embedding(x)
		h_embedding = self.embedding_dropout(h_embedding)
		h_lstm1, _ = self.lstm1(h_embedding)
		h_lstm2, _ = self.lstm2(h_lstm1)

		# attention
		# att = self.attention(x)

		# global average pooling
		avg_pool = torch.mean(h_lstm2, 1)

		# global max pooling
		max_pool, _ = torch.max(h_lstm2, 1)

		# concatenation
		h = torch.cat((max_pool, avg_pool), 1)

		h_linear1 = F.relu(self.linear1(h))
		h_linear2 = F.relu(self.linear2(h))

		out1 = F.sigmoid(self.linear_out(h_linear1))
		out2 = F.sigmoid(self.linear_aux_out(h_linear2))

		return out1, out2


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


### preprocessing
x_train = preprocess(train['comment_text'])
y_aux_train = np.array(train[['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']])

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# Overall
weights = np.ones((len(x_train),)) / 4
# Subgroup
weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

## loss_weight = 3.209226860170181
loss_weight = (1.0/weights.mean())*4/5

train['target'] = np.where(train['target'] >= 0.5, 1, 0)
# y_train = np.array(train['target']).reshape(-1,1)
y_train = np.vstack([train['target'],weights]).T
y_train = np.concatenate((y_train, y_aux_train), axis=1)
x_test = preprocess(test['comment_text'])


## Tokenize and padding
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train)+list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN,padding='post')

## Data Loader
test_data = TextDataset(x_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_fn_test)

#### Set model parameters
crawl_matrix ,_ = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
# glove_matrix ,_ = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
# embedding_matrix = np.concatenate((crawl_matrix, glove_matrix), axis=-1)

x_train_all = np.array(x_train)
y_train_all = np.array(y_train)

final_test = list()

skf = StratifiedKFold(n_splits=5,shuffle=True)

for index, (train_index, test_index) in enumerate(skf.split(x_train_all, y_train_all[:,0])):

	x_train, x_val = x_train_all[train_index], x_train_all[test_index]
	y_train, y_val = y_train_all[train_index], y_train_all[test_index]

	train_data = TextDataset(x_train, y_train)
	val_data = TextDataset(x_val, y_val)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_fn_train)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_fn_train)

	print("model: {}".format(index))
	net = NeuralNet(crawl_matrix,128)
	net.cuda()
	loss_fn = torch.nn.BCELoss(reduction='mean')
	optimizer = torch.optim.Adam(net.parameters())

	test_checkpoint = list()
	loss_checkpoint = list()

	for epoch in range(EPOCHS):  # loop over the dataset multiple times

		start_time = time.time()

		avg_loss = 0.0

		net.train()
		for i, data in enumerate(train_loader):

			# get the inputs
			inputs, labels = data

			label1 = labels[:,:2]
			label2 = labels[:,2:]

			## forward + backward + optimize
			pred1, pred2 = net(inputs)

			loss1 =	custom_loss(pred1, label1)
			loss2 = loss_fn(pred2,label2)
			loss = loss1*loss_weight+loss2

			# zero the parameter gradients
			optimizer.zero_grad()

			loss.backward()
			optimizer.step()

			avg_loss += loss.item()

		net.eval()

		avg_val_loss = 0.0

		for data in val_loader:

			# get the inputs
			inputs, labels = data

			val_label1 = labels[:,:2]
			val_label2 = labels[:,2:]

			## forward + backward + optimize
			pred1, pred2 = net(inputs)

			loss1_val = custom_loss(pred1, val_label1)
			# loss2_val = loss2 = loss_fn(pred2,val_label2)
			# loss_val = loss1_val*loss_weight+loss2_val

			avg_val_loss += loss1_val.item()

		elapsed_time = time.time() - start_time 

		print('Epoch {}/{} \t loss={:.4f}\t val_loss={:.4f} \t time={:.2f}s'.format(
						epoch+1, EPOCHS, avg_loss/len(train_loader),avg_val_loss/len(val_loader), elapsed_time))
		
		# print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
		# 				epoch+1, EPOCHS, avg_loss/len(train_loader), elapsed_time))

		## inference
		result = list()
		with torch.no_grad():
			for x_batch in test_loader:
				y_pred, _ = net(x_batch)
				y_pred = y_pred.cpu().numpy()[:,0]
				result.extend(y_pred)

		test_checkpoint.append(result)
		loss_checkpoint.append(avg_val_loss)

	final_test.append(test_checkpoint[np.argmin(loss_checkpoint)])

## submission
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = np.mean(final_test, axis=0)
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)



