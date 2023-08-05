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
    df = df[['postText', 'targetParagraphs', 'targetTitle']]

    # Convert all columns into strings
    df.loc[:, ['postText', 'targetParagraphs', 'targetTitle']] = df.loc[:, ['postText', 'targetParagraphs', 'targetTitle']].astype(str)

    # Tokenize the relevant columns (not actually used for the Bag of Word approach)
    tokenizer = RegexpTokenizer(r"\w+")
    df["postText_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["postText"]), axis=1)
    df["paragraph_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetParagraphs"]), axis=1)
    df["targetTitle_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetTitle"]), axis=1)

    # Removing stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    df["postText_tokens"] = df.apply(lambda row: [element for element in row["postText_tokens"] if element not in stopwords], axis=1)
    df["paragraph_tokens"] = df.apply(lambda row: [element for element in row["paragraph_tokens"] if element not in stopwords], axis=1)
    df["targetTitle_tokens"] = df.apply(lambda row: [element for element in row["targetTitle_tokens"] if element not in stopwords], axis=1)

    # Lowercasing
    df['postText_tokens'] = df['postText_tokens'].map(lambda row: list(map(str.lower, row)))
    df['paragraph_tokens'] = df['paragraph_tokens'].map(lambda row: list(map(str.lower, row)))
    df['targetTitle_tokens'] = df['targetTitle_tokens'].map(lambda row: list(map(str.lower, row)))

    # Multiple space to single space and remove special characters
    df[['postText_tokens', 'paragraph_tokens', 'targetTitle_tokens']] = df[['postText_tokens', 'paragraph_tokens', 'targetTitle_tokens']].replace(r'\s+', ' ', regex=True).replace(r'\W', ' ', regex=True)

    # Lemmatize tokens
    df['postText_tokens'] = df['postText_tokens'].apply(lemmatize_text)
    df['paragraph_tokens'] = df['paragraph_tokens'].apply(lemmatize_text)
    df['targetTitle_tokens'] = df['targetTitle_tokens'].apply(lemmatize_text)

    df['combined_texts'] = df['postText_tokens'].apply(lambda tokens: ' '.join(tokens)) + " " + df['paragraph_tokens'].apply(lambda tokens: ' '.join(tokens)) + " " + df['targetTitle_tokens'].apply(lambda tokens: ' '.join(tokens))

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
    return torch.tensor(embeddings, dtype=torch.float32)


# Define the Bi-LSTM neural network model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)  # Apply softmax activation
        return out


def main():
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

    hidden_size = 128
    output_size = 3  # Positive and negative classes
    num_layers = 2  # Number of LSTM layers
    learning_rate = 0.001

    model = BiLSTM(input_size, hidden_size, output_size, num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in tqdm(train_loader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
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

    model_filename = "model_bilstm.pth"
    torch.save(model.state_dict(), model_filename)

    model.load_state_dict(torch.load("model_bilstm.pth"))
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

    Out_df1.to_csv("output_bilstm_modified.csv", index=False)


if __name__ == "__main__":
    main()
