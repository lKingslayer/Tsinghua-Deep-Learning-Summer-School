# coding:utf-8
import logging

import torch
from torch import nn

# pylint: disable=W0221
class Network(nn.Module):
	def __init__(self, emb, rnn_size=200, mode='GRU'):
		super(Network, self).__init__()
		'''
		mode: 'GRU', 'LSTM', 'Attention'
		'''
		self.mode = mode

		self.embLayer = EmbeddingLayer(emb)
		self.encoder = Encoder(embedding_size=emb.shape[1], rnn_size=rnn_size, mode=mode)
		if mode == 'Attention':
			self.selfAttention = SelfAtt(hidden_size=rnn_size)
		self.predictionNetwork = PredictionNetwork(rnn_size=rnn_size)

		self.loss = nn.CrossEntropyLoss()

	def forward(self, sent, sent_length, label=None):

		embedding = self.embLayer.forward(sent)
		hidden_states = self.encoder.forward(embedding, sent_length)
		if self.mode == 'Attention':
			sentence_representation, penalization_loss = self.selfAttention.forward(hidden_states)
		else:
			sentence_representation = hidden_states.mean(dim=1)
		logit = self.predictionNetwork.forward(sentence_representation)

		if label is None:
			return logit

		classification_loss = self.loss(logit, label)
		if self.mode == 'Attention':
			return logit, classification_loss + penalization_loss * .0
		else:
			return logit, classification_loss

class EmbeddingLayer(nn.Module):
	def __init__(self, emb):
		super(EmbeddingLayer, self).__init__()

		vocab_size, embedding_size = emb.shape
		self.embLayer = nn.Embedding(vocab_size, embedding_size)
		self.embLayer.weight = nn.Parameter(torch.Tensor(emb))

	def forward(self, sent):
		'''
		inp: data
		output: post
		'''
		return self.embLayer(sent)

class LSTM(nn.Module):
	"""docstring for LSTM"""
	def __init__(self, input_size, hidden_size):
		super(LSTM, self).__init__()
		# TODO: Implement LSTM
		self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.memory_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.hidden_size = hidden_size
		# End TODO

	def forward(self, embedding, init_h=None, inin_c=None):
		'''
		embedding: [sentence_length, batch_size, embedding_size]
		init_h   : [batch_size, hidden_size]
		'''
		# TODO: Implement LSTM
		sentence_length, batch_size, embedding_size = embedding.size()
		if init_h is None:
			h = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			h = init_h
		hidden_state = []

		if inin_c is None:
			c = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			c = init_h
		self.last_c = c

		for t in range(sentence_length):
			_input = torch.cat([embedding[t], h], dim=1)
			f = torch.sigmoid(self.forget_gate(_input))
			i = torch.sigmoid(self.input_gate(_input))
			o = torch.sigmoid(self.output_gate(_input))
			c_hat = torch.tanh(self.memory_gate(_input))
			self.last_c = torch.add(torch.mul(f, self.last_c), torch.mul(i, c_hat))
			h = torch.mul(o, torch.tanh(self.last_c))
			hidden_state.append(h)

		return torch.stack(hidden_state, dim=1)
		# End TODO



class GRU(nn.Module):
	"""docstring for GRU"""
	def __init__(self, input_size, hidden_size):
		super(GRU, self).__init__()
		self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.memory_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.hidden_size = hidden_size

	def forward(self, embedding, init_h=None):
		'''
		embedding: [sentence_length, batch_size, embedding_size]
		init_h   : [batch_size, hidden_size]
		'''
		sentence_length, batch_size, embedding_size = embedding.size()
		if init_h is None:
			h = torch.zeros(batch_size, self.hidden_size, \
							dtype=embedding.dtype, device=embedding.device)
		else:
			h = init_h
		hidden_states = []
		for t in range(sentence_length):
			_input = torch.cat([embedding[t], h], dim=1)
			z = torch.sigmoid(self.update_gate(_input)) # [batch_size, hidden_size]
			r = torch.sigmoid(self.reset_gate(_input)) # [batch_size, hidden_size]
			# TODO: Update hidden state h
			_input = torch.cat([embedding[t], torch.mul(r, h)], dim=1)
			h_hat = torch.tanh(self.memory_gate(_input))
			h = torch.add(torch.mul((1-z), h), torch.mul(z, h_hat)) # [batch_size, hidden_size]
			hidden_states.append(h)
			# End TODO

		return torch.stack(hidden_states, dim=1) # [batch_size, sentence_length, hidden_size]

class Encoder(nn.Module):
	def __init__(self, embedding_size, rnn_size, mode='GRU'):
		super(Encoder, self).__init__()

		if mode == 'GRU':
			self.rnn = GRU(embedding_size, rnn_size)
		else:
			self.rnn = LSTM(embedding_size, rnn_size)

	def forward(self, embedding, sent_length=None):
		'''
		sent_length is not used
		'''
		hidden_states = self.rnn(embedding.transpose(0, 1)) # [batch_size, sentence_length, hidden_size]
		# you can add dropout here
		# dropout(0.15)
		# End
		return hidden_states

class SelfAtt(nn.Module):
	"""docstring for SelfAtt"""
	def __init__(self, hidden_size):
		super(SelfAtt, self).__init__()
		# TODO: Implement Self-Attention
		self.hidden_size = hidden_size


	def forward(self, h, add_penalization=True):
		'''
		h: [batch_size, sentence_length, hidden_size]
		'''
		# TODO: Implement Self-Attention
		H = h
		bs = H.size(0)  # batch size
		sentence_size = H.size(1) # sentence_size
		hidden_size = H.size(2) # hidden_size
		r = 1 # hiperparameter
		f_1 = nn.Linear(hidden_size, hidden_size, bias=False).cuda()
		w_1 = torch.tanh(f_1(H))
		f_2 = nn.Linear(hidden_size, r, bias=False).cuda()
		A = torch.softmax(f_2(w_1), dim=1)
		AT = torch.transpose(A, 1, 2)
		M = torch.bmm(AT, H)
		#print("M",M.size())

		M = torch.squeeze(M, dim = 1)
		AAT = torch.bmm(AT, A)
		I = torch.eye(r).cuda()
		per_term = torch.norm(AAT-I)
		return M, torch.square(per_term)

		# End TODO


class PredictionNetwork(nn.Module):
	def __init__(self, rnn_size, hidden_size=64, class_num=5):
		super(PredictionNetwork, self).__init__()
		self.predictionLayer = nn.Sequential(nn.Linear(rnn_size, hidden_size),
											nn.ReLU(),
											nn.Linear(hidden_size, class_num))

	def forward(self, h):

		return self.predictionLayer(h)
