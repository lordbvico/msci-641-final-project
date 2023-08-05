import numpy as np
import pandas as pd
import sys
import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

sweep_config = {
        "name": "sweep",
        "method": "random",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "num_filters": {"values": [64, 128, 256]},
            "kernel_sizes": {"values": [[2, 3, 4], [3, 4, 5], [4, 5, 6]]},
            'learning_rate': {'max': 0.1, 'min': 0.0001},
            "num_epochs": {"values": [1,2,5,7,10]},
            "lstm_hidden_size": {"values": [64, 128, 256]},
            "lstm_num_layers": {"values": [1,2]}            
        }
    }

    # Initialize Wandb sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="CNN LSTM Parameter Sweep Set 2")

train_data_path = "data/train.jsonl"
valid_data_path = "data/val.jsonl"
test_data_path = "data/test.jsonl"

df = pd.read_json(train_data_path, lines=True)
df2 = pd.read_json(valid_data_path, lines=True)
df3 = pd.read_json(test_data_path, lines=True)
concatenated_df = pd.concat([df, df2], ignore_index=True)


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


def preprocess_data(df):
    df = df[['postText', 'targetParagraphs']]

    # Convert all columns into strings
    df.loc[:, ['postText', 'targetParagraphs']] = df.loc[:, ['postText', 'targetParagraphs']].astype(str)

    # Tokenize the relevant columns (not actually used for the Bag of Word approach)
    tokenizer = RegexpTokenizer(r"\w+")
    df["postText_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["postText"]), axis=1)
    df["paragraph_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetParagraphs"]), axis=1)
   # df["targetTitle_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetTitle"]), axis=1)

    # Removing stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    df["postText_tokens"] = df.apply(lambda row: [element for element in row["postText_tokens"] if element not in stopwords], axis=1)
    df["paragraph_tokens"] = df.apply(lambda row: [element for element in row["paragraph_tokens"] if element not in stopwords], axis=1)
  #  df["targetTitle_tokens"] = df.apply(lambda row: [element for element in row["targetTitle_tokens"] if element not in stopwords], axis=1)

    # Lowercasing
    df['postText_tokens'] = df['postText_tokens'].map(lambda row: list(map(str.lower, row)))
    df['paragraph_tokens'] = df['paragraph_tokens'].map(lambda row: list(map(str.lower, row)))
   # df['targetTitle_tokens'] = df['targetTitle_tokens'].map(lambda row: list(map(str.lower, row)))

    # Multiple space to single space and remove special characters
    df[['postText_tokens', 'paragraph_tokens']] = df[['postText_tokens', 'paragraph_tokens']].replace(r'\s+', ' ', regex=True).replace(r'\W', ' ', regex=True)

    # Lemmatize tokens
    df['postText_tokens'] = df['postText_tokens'].apply(lemmatize_text)
    df['paragraph_tokens'] = df['paragraph_tokens'].apply(lemmatize_text)
   # df['targetTitle_tokens'] = df['targetTitle_tokens'].apply(lemmatize_text)

    df['combined_texts'] = df['postText_tokens'].apply(lambda tokens: ' '.join(tokens)) + " " + df['paragraph_tokens'].apply(lambda tokens: ' '.join(tokens))

    return df


def convert_text_to_embeddings(text_data, word_embeddings, max_seq_length):
    embeddings = []
    for text in text_data:
        sentence_embedding = np.zeros((max_seq_length, word_embeddings.vector_size))
        count = 0
        for i, word in enumerate(text):
            if i < max_seq_length and word in word_embeddings:
                sentence_embedding[i] = word_embeddings[word]
                count += 1
        if count > 0:
            sentence_embedding /= count
        embeddings.append(sentence_embedding)
    #return torch.tensor(embeddings, dtype=torch.float32)
    return torch.tensor(embeddings, dtype=torch.float32).permute(0, 2, 1)


# Define the CNN neural network model
class CNNLSTM(nn.Module):
    def __init__(self, input_size, num_filters, kernel_sizes, lstm_hidden_size, lstm_num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([nn.Conv1d(input_size, num_filters, kernel_size) for kernel_size in kernel_sizes])

        lstm_input_size = len(kernel_sizes) * num_filters
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # CNN for local feature extraction
        x = x.permute(0, 2, 1)  # Reshape input for Conv1d
        conv_outputs = []
        for conv in self.convs:
            conv_out = nn.functional.relu(conv(x))
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        x = torch.cat(conv_outputs, 1)

      #  print("CNN output shape:", x.shape)  # Add this line to print CNN output shape

        # LSTM for sequential processing
        lstm_out, _ = self.lstm(x)

      #  print("LSTM output shape:", lstm_out.shape)  # Add this line to print LSTM output shape

        lstm_out = lstm_out.unsqueeze(1) # Add a new dimension with size 1
        
        # Select the last time step's output (LSTM's output is sorted by sequence length, we select the last one)
        lstm_out = lstm_out[:, -1, :]

        # Concatenate CNN and LSTM outputs
        combined_out = lstm_out

        out = self.fc(combined_out)
        out = nn.functional.softmax(out, dim=1)  # Apply softmax activation
        return out

def main():
    run = wandb.init()
    train_selected = preprocess_data(df)
    val_selected = preprocess_data(df2)
    test_selected = preprocess_data(df3)
    word_vec_selected = preprocess_data(concatenated_df)

    tokenizer = RegexpTokenizer(r"\w+")
    word_vec_selected['combined_texts_tokens'] = word_vec_selected.apply(lambda row: tokenizer.tokenize(row["combined_texts"]), axis=1)
    word_train = word_vec_selected['combined_texts_tokens']
    train_selected['combined_texts_tokens'] = train_selected.apply(lambda row: tokenizer.tokenize(row["combined_texts"]), axis=1)
    val_selected['combined_texts_tokens'] = val_selected.apply(lambda row: tokenizer.tokenize(row["combined_texts"]), axis=1)
    test_selected['combined_texts_tokens'] = test_selected.apply(lambda row: tokenizer.tokenize(row["combined_texts"]), axis=1)
    X_train = train_selected['combined_texts_tokens']
    X_val = val_selected['combined_texts_tokens']
    X_test = test_selected['combined_texts_tokens']

    label_encoder = LabelEncoder()

    train_labels = [tag for tags in df['tags'] for tag in tags]

    label_encoder.fit(train_labels)

    train_y = [label_encoder.transform(tags) for tags in df['tags']]
    val_y = [label_encoder.transform(tags) for tags in df2['tags']]

    train_y = [label_arr.ravel() for label_arr in train_y]
    val_y = [label_arr.ravel() for label_arr in val_y]
    train_y = [item[0] for item in train_y]
    train_y = np.array(train_y)
    val_y = [item[0] for item in val_y]
    val_y = np.array(val_y)

    word2vec_model = Word2Vec(sentences=word_train, vector_size=100, min_count=1)

    # Load your pre-trained word2vec embeddings
    word_embeddings = word2vec_model.wv
    input_size = word_embeddings.vector_size
    max_seq_length = 100
    train_features = convert_text_to_embeddings(X_train, word_embeddings, max_seq_length)
    train_labels = torch.tensor(train_y, dtype=torch.long)
    val_features = convert_text_to_embeddings(X_val, word_embeddings, max_seq_length)
    val_labels = torch.tensor(val_y, dtype=torch.long)
    test_features = convert_text_to_embeddings(X_test, word_embeddings, max_seq_length)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    #num_filters = 128
    #kernel_sizes = [2, 3, 4]
    #lstm_hidden_size = 64
    #lstm_num_layers = 1
    output_size = len(label_encoder.classes_)  # Number of classes for classification
    #learning_rate = 0.001
    lstm_hidden_size = wandb.config.lstm_hidden_size
    lstm_num_layers = wandb.config.lstm_num_layers
    num_filters = wandb.config.num_filters
    kernel_sizes = wandb.config.kernel_sizes
    learning_rate = wandb.config.learning_rate
    
    
    model = CNNLSTM(input_size=word_embeddings.vector_size, num_filters=num_filters,
                    kernel_sizes=kernel_sizes, lstm_hidden_size=lstm_hidden_size,
                    lstm_num_layers=lstm_num_layers, output_size=output_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #num_epochs = 10
    num_epochs = wandb.config.num_epochs
    best_val_accuracy = 0.0
    best_model_state = None   
   
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in tqdm(train_loader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
         #   print("Batch features shape:", batch_features.shape)  # Add this line to print batch features shape
          #  print("Batch labels shape:", batch_labels.shape)      # Add this line to print batch labels shape
            
            
            optimizer.zero_grad()
            outputs = model(batch_features)
          #  print("Outputs shape:", outputs.shape)                # Add this line to print outputs shape
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for val_features, val_labels in tqdm(val_loader):
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_features)
                val_loss += criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_predicted = val_predicted.cpu()
                val_labels = val_labels.cpu()
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print()
        
             
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
        wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_accuracy, "Kernel Size": kernel_sizes, "Learning Rate": learning_rate, 
                   "Number of Filters": num_filters, "Number of Epochs": num_epochs, "LSTM Hidden Size": lstm_hidden_size, "LSTM Num Layers": lstm_num_layers})   
   
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    best_model_filename = "best_model_cnn_lstm_wandb.pth"
    torch.save(best_model_state, best_model_filename)

    model.load_state_dict(torch.load(best_model_filename))
    model.eval()
    test_features = test_features.to(device)
    test_outputs = model(test_features)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_predicted = test_predicted.cpu()
    converted_array = test_predicted.numpy()

    converted_array_flattened = converted_array.ravel()
    converted_array_tags = label_encoder.inverse_transform(converted_array_flattened)

    output_data1 = []
    for index, pred in enumerate(converted_array_tags):
        prediction1 = {'id': index, 'spoilerType': pred}
        output_data1.append(prediction1)
    Out_df1 = pd.DataFrame(output_data1)

    Out_df1.to_csv("output_cnn_lstm_wandb_best.csv", index=False)


if __name__ == "__main__":
    wandb.agent(sweep_id, function=main)