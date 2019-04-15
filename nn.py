import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from keras.preprocessing import text, sequence
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
BATCH_SIZE = 2048
EPOCHS = 10
MAX_LEN = 220
max_feature = 100000
embedding_size = 300

def preprocess(data):

	'''
	Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
	'''
	punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
	def clean_special_chars(text, punct):
			for p in punct:
					text = text.replace(p, ' ')
			return text

	data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
	return data

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, path):

	"""
	https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
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

class NeuralNet(nn.Module):

	def __init__(self,embedding_matrix,num_unit):
		super(NeuralNet, self).__init__()

		self.embedding = nn.Embedding(max_feature, embedding_size)
		self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
		self.embedding.weight.requires_grad = False
		self.lstm1 = nn.LSTM(embedding_size, num_unit, bidirectional=True, batch_first=True)
		self.lstm2 = nn.LSTM(num_unit*2, num_unit, bidirectional=True, batch_first=True)
		self.linear = nn.Linear(num_unit*4, 1)

	def forward(self, x):

		x = self.embedding(x)
		x, _ = self.lstm1(x)
		x, _ = self.lstm2(x)

		# global average pooling
		avg_pool = torch.mean(x, 1)

		# global max pooling
		max_pool, _ = torch.max(x, 1)

		h = torch.cat((max_pool, avg_pool), 1)
		out = F.sigmoid(self.linear(h))
		return out

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

### preprocessing
x_train = preprocess(train['comment_text'])
# y_train = np.where(train['target'] >= 0.5, 1, 0).reshape(-1,1)
y_train = np.array(train['target']).reshape(-1,1)
# y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = preprocess(test['comment_text'])

tokenizer = text.Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(list(x_train)+list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN,padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN,padding='post')


#### Set model parameters
x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test_data = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

embedding_matrix ,_ = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

net = NeuralNet(embedding_matrix,128)
net.cuda()

loss_fn = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters())

train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

x_train = torch.tensor(x_train, dtype=torch.long).cuda()
x_val = torch.tensor(x_val, dtype=torch.long).cuda()
y_train = torch.tensor(y_train, dtype=torch.float).cuda()
y_val = torch.tensor(y_val, dtype=torch.float).cuda()

train_data = torch.utils.data.TensorDataset(x_train, y_train)
val_data = torch.utils.data.TensorDataset(x_val, y_val)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


for epoch in range(EPOCHS):  # loop over the dataset multiple times

	start_time = time.time()
	avg_loss = 0.0
	net.train()

	for i, data in enumerate(train_loader):

		# get the inputs
		inputs, labels = data

		## forward + backward + optimize
		pred = net(inputs)

		loss = loss_fn(pred,labels)

		# zero the parameter gradients
		optimizer.zero_grad()

		loss.backward()
		optimizer.step()
		# print("loss:{:.4f}".format(loss.item()))

		avg_loss += loss.item()

	net.eval()
	avg_val_loss = 0.0

	for data in val_loader:
		inputs, labels = data
		pred = net(inputs)
		loss = loss_fn(pred,labels)
		avg_val_loss += loss.item()

	elapsed_time = time.time() - start_time 
	print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
					epoch+1, EPOCHS, avg_loss/len(train_loader),avg_val_loss/len(val_loader), elapsed_time))


print('Finished Training')

## inference
result = list()
with torch.no_grad():
	for (x_batch,) in test_loader:
		y_pred = net(x_batch).cpu().numpy().reshape(-1)
		result.extend(y_pred)

## submission
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission['prediction'] = result
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)




