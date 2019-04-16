import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import os
import itertools
import matplotlib.pyplot as plt
import pickle
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import model as m

############# Parameters
path2embeddings = '../pretrained_embeds/glove.6B/'
embedfile = 'glove.6B.50d'

path2data = '../data/'
datafile = 'news_articles.pkl'

cat2id_file = 'category2id.pkl'

model_saving_path = '../attention/'
model_saving_file = 'attention_model.pt'

article_max_len = 600

lr = 1e-4
epochs = 100
embed_size = 50
hidden_dim = 100
n_classes = 7



############# Loading Pretrained Glove Embeddings
if os.path.isfile(path2embeddings + embedfile + '_w2v.txt'):
    glove_model = KeyedVectors.load_word2vec_format(path2embeddings + embedfile + '_w2v.txt', binary=False)
else:
    glove2word2vec(glove_input_file=path2embeddings + embedfile + '.txt', word2vec_output_file=path2embeddings + embedfile + '_w2v.txt')
    glove_model = KeyedVectors.load_word2vec_format(path2embeddings + embedfile + '_w2v.txt', binary=False)

def get_embed(word):
    # Case folding
    word = word.lower()
    try:
        return (glove_model.get_vector(word))
    except:
        return (glove_model.get_vector('<unk>'))



############# Loading the Data
data_df = pd.read_pickle(path2data + datafile)
data = data_df[data_df['source_article']!='NA'][['source_article', 'category']]
data = data.sample(frac=1).reset_index(drop=True)
train, validate, test = np.split(data.sample(frac=1), [int(.8*len(data)), int(.9*len(data))])


## Categories to its id
if os.path.exists(path2data + cat2id_file):
	with open(path2data + cat2id_file, 'rb') as handle:
		category2id = pickle.load(handle)
else:
	category2id = {}
	count = 0
	for c in set(data['category'].tolist()):
		category2id[c] = count
		count += 1

	with open(path2data + cat2id_file, 'wb') as handle:
		pickle.dump(category2id, handle, protocol=pickle.HIGHEST_PROTOCOL)




############ Defining the model
# Using gpu if available else cpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = m.atten_classifier(embed_size, hidden_dim, n_classes)
model = model.to(device)

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)




############# Helper function
stop = stopwords.words('english') + list(string.punctuation)

def single_epoch(data, model, optimizer, criterion, purpose = 'Train'):
	model_output = []
	true_output = []
	alphas_list = []

	total_loss = 0
	datapoints = 0
	avg_loss = 0

	for index, row in data.iterrows():

		# Pre-processing
		article = row['source_article']
		article = article.lower()
		article = nltk.word_tokenize(article)
		article = [get_embed(i) for i in article if i not in stop]
		article = np.array(article[:article_max_len])
		article_domain = category2id[row['category']]

		article_inp = torch.from_numpy(article).to(device)
		domain_out = torch.tensor([article_domain]).to(device)

		out, alphas = model(article_inp)
		# Append alphas to the list
		alpha_list = alphas.data.cpu().numpy().reshape(-1).tolist()
		alphas_list.append(alpha_list)

		if purpose == 'Train' or purpose == 'Validate':
			optimizer.zero_grad()
			# Calculate loss 
			loss = criterion(out, domain_out)
			# Accumulating the losses
			loss_data = float(loss.data.cpu().numpy())
			total_loss += loss_data

			if purpose == 'Train':
				# Update only at training time and not during validation
				loss.backward()
				optimizer.step()


			datapoints += 1
			avg_loss = total_loss/datapoints

		# Getting output domain class
		domain_class = int(torch.max(out, 1)[1].cpu().numpy()[-1])
		model_output.append(domain_class)
		true_output.append(article_domain)

	return model, avg_loss, alphas_list, model_output, true_output



######### Training Loop
prev_validate_loss = 10000 # infinite loss

for e in range(epochs):
	model, avg_loss, alphas_list, pred, true_output = single_epoch(train, model, optimizer, criterion, purpose = 'Train')
	print('Epoch: ' + str(e) + '/' + str(epochs) + ', Loss: ' + str(avg_loss))

	model, validate_loss, alphas_list, pred, true_output = single_epoch(validate, model, optimizer, criterion, purpose = 'Validate')
	print('Epoch: ' + str(e) + '/' + str(epochs) + ', Loss: ' + str(validate_loss))

	model, loss, alphas_list, pred, true_output = single_epoch(test, model, optimizer, criterion, purpose = 'Test')
	print('Accuracy on test set: ' + str(accuracy_score(true_output, pred)) + '\n\n')


	if validate_loss < prev_validate_loss:
		print('Model has improved, Saving!')
		torch.save(model.state_dict(), model_saving_path + model_saving_file)
		prev_validate_loss = validate_loss
		best_acc = str(accuracy_score(true_output, pred))


print(best_acc)
