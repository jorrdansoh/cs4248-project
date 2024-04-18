import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np

from model import TransformerClassifier

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
		text = re.sub(r'\d+', '', text)  # Remove numbers
		text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
		text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
		text = text.split()  # Tokenize into words
		text = [word for word in text if word not in stop_words]  # Remove stopwords
		text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize words
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
		
		return {
			'text': original_text,  # Add the original text here
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': 1 - encoding['attention_mask'].flatten(),
			'labels': torch.tensor(label, dtype=torch.long)
		}

if __name__ == "__main__":
	# Initialize the BERT tokenizer or any tokenizer that suits your model's architecture
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# Let's read the CSV file and define the texts and labels variables accordingly.

	# Load the dataset
	data = pd.read_csv("./raw_data/fulltrain.csv")

	# Split the dataframe into texts and labels
	texts = data.iloc[:, 1]  # Assuming the text data is in the second column
	labels = data.iloc[:, 0] - 1  # convert 1, 2, 3, 4 to 0, 1, 2, 3

	# Split the data into training and validation sets
	texts_train, texts_val, labels_train, labels_val = train_test_split(
		texts, labels, test_size=0.2, random_state=26
	)

	# Reset index to ensure alignment
	texts_train = texts_train.reset_index(drop=True)
	labels_train = labels_train.reset_index(drop=True)
	texts_val = texts_val.reset_index(drop=True)
	labels_val = labels_val.reset_index(drop=True)

	# Compute class weights based on the training set only
	# class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
	# class_weights = torch.tensor(class_weights, dtype=torch.float)
	# class_weights = torch.tensor([0.8721, 1.7573, 0.6829, 1.2195])
	class_weights = torch.tensor([1, 1.5, 1, 1.5])
	# print(labels.value_counts())
	# print("Class weights:", class_weights)

	# Compute sample weights for the training set
	sample_weights = np.array([class_weights[label] for label in labels_train])
	sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

	# Create the dataset
	train_dataset = TextDataset(texts_train, labels_train, tokenizer)
	val_dataset = TextDataset(texts_val, labels_val, tokenizer)

	# Create the data loaders
	batch_size = 32  # Adjust based on your system's capabilities
	train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	# Assuming the TransformerClassifier class is already defined as per your previous input
	input_dim = len(tokenizer.vocab)  # This should match the vocabulary size of the tokenizer
	output_dim = 4  # Number of classes
	model = TransformerClassifier(input_dim=input_dim, output_dim=output_dim)

	# Move the model to GPU if available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)

	# Define the optimizer and the loss function
	optimizer = optim.Adam(model.parameters())
	loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

	predictions_file_path = 'predictions.csv'

	# Training loop
	num_epochs = 50  # You can adjust the number of epochs
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		model.train()
		total_loss = 0
		i = 0
		for batch in train_loader:
			i += 1
			optimizer.zero_grad()
			
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device).to(torch.bool)
			labels = batch['labels'].to(device)
			
			# Forward pass
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			print('Batch {}'.format(i))
			
			# Compute loss
			loss = loss_fn(outputs, labels)
			total_loss += loss.item()
			
			# Backward pass and optimize
			loss.backward()
			# for name, parameter in model.named_parameters():
			# 	if parameter.grad is not None:
			# 		print(f"{name} gradient norm: {parameter.grad.data.norm(2).item()}")

			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
		
		print('Epoch {}/{}, Loss: {}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))

		# Validation step
		model.eval()
		total_val_loss = 0
		true_labels = []
		predictions = []
		text_sentences = []

		with torch.no_grad():
			for batch in val_loader:
				input_ids = batch['input_ids'].to(device)
				attention_mask = batch['attention_mask'].to(device).to(torch.bool)
				labels = batch['labels'].to(device)
				# print(labels + 1)
				text = batch['text']

				outputs = model(input_ids=input_ids, attention_mask=attention_mask)
				loss = loss_fn(outputs, labels)
				total_val_loss += loss.item()

				# Get the predictions and true labels for F1 score calculation
				_, predicted_labels = torch.max(outputs, dim=1)
				print(predicted_labels + 1)
				predictions.extend(predicted_labels.cpu().numpy())
				true_labels.extend(labels.cpu().numpy())
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
