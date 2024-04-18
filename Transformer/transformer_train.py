import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BertTokenizer
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import os
import numpy as np

from transformer_model import Transformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
	
class TextDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_length=512):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length

	def preprocess_text(self, text):
		text = text.lower()  # Lowercase text
		text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
		text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
		text = text.split()  # Tokenize into words
		text = [word for word in text if word not in stop_words]  # Remove stopwords
		#text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize words
		return ' '.join(text)  # Join words back into a single string
	
	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		original_text = self.texts.iloc[idx]
		label = self.labels[idx]
		processed_text = self.preprocess_text(original_text)
		
		encoding = self.tokenizer.encode_plus(
			processed_text,
			add_special_tokens=True,
			max_length=self.max_length,
			return_token_type_ids=False,
			padding='max_length',
			truncation=True,
			return_attention_mask=True,
			return_tensors='pt',
		)

		return encoding['input_ids'].flatten(), torch.tensor(label, dtype=torch.long), original_text
		
	

def data_init():
	train_path = "C:/Users/darre/OneDrive/Desktop/CS4248/Project/updated/train.csv"
	val_path = "C:/Users/darre/OneDrive/Desktop/CS4248/Project/updated/val.csv"
	test_path = "C:/Users/darre/OneDrive/Desktop/CS4248/Project/updated/balancedtest.csv"

	train_data = pd.read_csv(train_path)
	val_data = pd.read_csv(val_path)
	test_data = pd.read_csv(test_path, header=None)

	train_texts = train_data.iloc[:, 1]
	train_labels = train_data.iloc[:, 0].values - 1

	val_texts = val_data.iloc[:, 1]
	val_labels = val_data.iloc[:, 0].values - 1

	test_texts = test_data.iloc[:, 1]
	test_labels = test_data.iloc[:, 0].values - 1

	return train_texts,train_labels,val_texts,val_labels,test_texts,test_labels



if __name__ == "__main__":
	# Initialize the BERT tokenizer or any tokenizer that suits your model's architecture
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	#load dataset
	train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = data_init()

	# Return the lengths to confirm the variables are defined correctly
	print((len(train_texts), len(train_labels)), (len(val_texts), len(val_labels)), (len(test_texts),len(test_labels)))

	# Compute class weights based on the training set only
	class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
	class_weights = torch.tensor(class_weights, dtype=torch.float)

	# Compute sample weights for the training set
	sample_weights = np.array([class_weights[label] for label in train_labels])
	sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

	# Create the dataset
	train_dataset = TextDataset(train_texts, train_labels, tokenizer)
	val_dataset = TextDataset(val_texts, val_labels, tokenizer)
	test_dataset = TextDataset(test_texts, test_labels, tokenizer)

	# Create the data loaders
	batch_size = 32  # Adjust based on your system's capabilities
	train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
	#train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

	input_dim = len(tokenizer.vocab)
	output_dim = 4 
	model = Transformer(vocab_size = input_dim)

	# Move the model to GPU if available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)

	print(device)

	# Define the optimizer and the loss function
	optimizer = optim.Adam(model.parameters())
	loss_fn = nn.CrossEntropyLoss()

	predictions_file_path = 'C:/Users/darre/OneDrive/Desktop/CS4248/Project/updated/predictions.csv'
	test_file_path = 'C:/Users/darre/OneDrive/Desktop/CS4248/Project/updated/test_predictions.csv'

	train_f1_scores = []
	val_f1_scores = []
	test_f1_scores = []

	# Training loop
	num_epochs = 10
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		model.train()
		total_loss = 0
		i = 0
		total_f1 = 0
		train_predictions = []
		train_true_labels = []
		for (inputs, targets, text) in train_loader:
			i += 1
			optimizer.zero_grad()

			inputs = inputs.to(device)
			targets = targets.to(device)
			
			# Forward pass
			outputs = model(inputs)
			
			# Compute loss
			loss = loss_fn(outputs, targets)
			total_loss += loss.item()

			# Get the predictions and true labels for F1 score calculation
			_, predicted_labels = torch.max(outputs, dim=1)

			train_predictions.extend(predicted_labels.cpu().numpy())
			train_true_labels.extend(targets.cpu().numpy())

			
			
			# Backward pass and optimize
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
		
		f1 = f1_score(train_true_labels, train_predictions, average='weighted')
		print('Epoch {}/{}, Loss: {}, F1 Score: {}'.format(epoch + 1, num_epochs, total_loss / len(train_loader), f1))

		# Validation step
		model.eval()
		total_val_loss = 0
		true_labels = []
		predictions = []
		text_sentences = []

		with torch.no_grad():
			for inputs, targets, text in val_loader:
				inputs = inputs.to(device)
				targets = targets.to(device)
				outputs = model(inputs)
				loss = loss_fn(outputs, targets)
				total_val_loss += loss.item()

				# Get the predictions and true labels for F1 score calculation
				_, predicted_labels = torch.max(outputs, dim=1)
				predictions.extend(predicted_labels.cpu().numpy())
				true_labels.extend(targets.cpu().numpy())
				text_sentences.extend(text)
		
		# Calculate F1 score
		f1 = f1_score(true_labels, predictions, average='weighted')

		# Print validation results
		print('Epoch {}/{}, Validation Loss: {}, F1 Score: {}'.format(epoch + 1, num_epochs, total_val_loss / len(val_loader), f1))

		# Write text sentences and predictions to CSV file
		with open(predictions_file_path, 'w', newline='', encoding='utf-8') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for sentence, pred in zip(text_sentences, predictions):
				writer.writerow([pred + 1, sentence])

		# Print that the predictions were saved
		print('Predictions for epoch {} were saved to {}'.format(epoch + 1, predictions_file_path))
	
	model.eval()
	print("Testing...")

	total_test_loss = 0
	true_test_labels = []
	predictions_test = []
	text_sentences_test = []

	with torch.no_grad():
		for inputs, targets, text in test_loader:
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = model(inputs)
			loss = loss_fn(outputs, targets)
			total_test_loss += loss.item()

			# Get the predictions and true labels for F1 score calculation
			_, predicted_labels = torch.max(outputs, dim=1)
			predictions_test.extend(predicted_labels.cpu().numpy())
			true_test_labels.extend(targets.cpu().numpy())
			text_sentences_test.extend(text)
		
	# Calculate F1 score
	f1_test = f1_score(true_test_labels, predictions_test, average='weighted')
	print('Test Loss: {}, Test F1 Score: {}'.format( total_test_loss / len(test_loader), f1_test))

	# Write text sentences and predictions to CSV file
	with open(test_file_path, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for sentence, pred in zip(text_sentences_test, predictions_test):
			writer.writerow([pred + 1, sentence])

	# Print that the predictions were saved
	print('Predictions for epoch {} were saved to {}'.format(epoch + 1, test_file_path))


