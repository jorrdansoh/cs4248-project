import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

block_size = 200
embeds_size = 100
num_classes = 4
drop_prob = 0.13
num_heads = 4
head_size = embeds_size // num_heads


class Block(nn.Module):
	def __init__(self):
		super(Block, self).__init__()
		self.attention = nn.MultiheadAttention(embeds_size, num_heads, batch_first=True)
		self.ffn = nn.Sequential(
			nn.Linear(embeds_size, 2 * embeds_size),
			nn.LeakyReLU(),
			nn.Linear(2 * embeds_size, embeds_size),
		)
		self.drop1 = nn.Dropout(drop_prob)
		self.drop2 = nn.Dropout(drop_prob)
		self.ln1 = nn.LayerNorm(embeds_size)
		self.ln2 = nn.LayerNorm(embeds_size)

	def forward(self, hidden_state):
		attn, _ = self.attention(hidden_state, hidden_state, hidden_state, need_weights=False)
		attn = self.drop1(attn)
		out = self.ln1(hidden_state + attn)
		observed = self.ffn(out)
		observed = self.drop2(observed)
		return self.ln2(out + observed)


class Transformer(nn.Module):
	def __init__(self, vocab_size):
		super(Transformer, self).__init__()

		self.tok_emb = nn.Embedding(vocab_size, embeds_size)
		self.block = Block()
		self.ln1 = nn.LayerNorm(embeds_size)
		self.ln2 = nn.LayerNorm(embeds_size)

		self.classifier_head = nn.Sequential(
			nn.Linear(embeds_size, embeds_size),
			nn.LeakyReLU(),
			nn.Dropout(drop_prob),
			nn.Linear(embeds_size, embeds_size),
			nn.LeakyReLU(),
			nn.Linear(embeds_size, num_classes),
			nn.Softmax(dim=1),
		)

	def num_params(self):
		n_params = sum(p.numel() for p in self.parameters())
		return n_params

	def forward(self, seq):
		embedded = self.tok_emb(seq)
		output = self.block(embedded)
		output = output.mean(dim=1)
		output = self.classifier_head(output)
		return output
