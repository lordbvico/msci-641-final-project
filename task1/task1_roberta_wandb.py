import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score


train_data_path = "data/train.jsonl"
valid_data_path = "data/val.jsonl"
test_data_path = "data/test.jsonl"

df = pd.read_json(train_data_path, lines=True)
df2 = pd.read_json(valid_data_path, lines=True)
df3 = pd.read_json(test_data_path, lines=True)
concatenated_df = pd.concat([df, df2], ignore_index=True)


def preprocess_data(df):
    df = df[['postText', 'targetParagraphs', 'targetTitle']]

    # Convert all columns into strings
    df.loc[:, ['postText', 'targetParagraphs', 'targetTitle']] = df.loc[:, ['postText', 'targetParagraphs', 'targetTitle']].astype(str)

    # Concatenate text columns
    df['combined_texts'] = df['postText'] + " " + df['targetParagraphs'] + " " + df['targetTitle']

    return df


def encode_labels(train_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    encoded_labels = label_encoder.transform(train_labels)
    return encoded_labels, label_encoder


def encode_texts(texts, tokenizer, max_seq_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids.append(encoding['input_ids'].squeeze())
        attention_masks.append(encoding['attention_mask'].squeeze())

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    return input_ids, attention_masks


class RoBERTaClassifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


def train_model(config):
    train_selected = preprocess_data(df)
    val_selected = preprocess_data(df2)
    test_selected = preprocess_data(df3)

    X_train = train_selected['combined_texts']
    X_val = val_selected['combined_texts']
    X_test = test_selected['combined_texts']

    train_labels, label_encoder = encode_labels(df['tags'])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    input_size = tokenizer.vocab_size
    max_seq_length = config.max_seq_length

    train_input_ids, train_attention_masks = encode_texts(X_train, tokenizer, max_seq_length)
    val_input_ids, val_attention_masks = encode_texts(X_val, tokenizer, max_seq_length)
    test_input_ids, test_attention_masks = encode_texts(X_test, tokenizer, max_seq_length)

    train_labels = torch.tensor(train_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    hidden_size = 768
    output_size = len(label_encoder.classes_)
    learning_rate = config.learning_rate

    model = RoBERTaClassifier(hidden_size, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    num_epochs = config.num_epochs

    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_masks, labels in tqdm(train_loader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for input_ids, attention_masks in tqdm(val_loader):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                logits = model(input_ids, attention_masks)
                _, predicted = torch.max(logits, 1)
                val_correct += predicted.size(0)

        val_accuracy = val_correct / len(val_loader.dataset)

        wandb.log({'epoch': epoch + 1, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

    model_filename = "model_roberta.pth"
    torch.save(model.state_dict(), model_filename)

    model.load_state_dict(torch.load(model_filename))
    model.eval()
    test_input_ids = test_input_ids.to(device)
    test_attention_masks = test_attention_masks.to(device)
    test_outputs = model(test_input_ids, test_attention_masks)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_predicted = test_predicted.cpu().numpy()

    converted_array_flattened = test_predicted.ravel()
    converted_array_tags = label_encoder.inverse_transform(converted_array_flattened)

    output_data1 = []
    for index, pred in enumerate(converted_array_tags):
        prediction1 = {'id': index, 'spoilerType': pred}
        output_data1.append(prediction1)
    Out_df1 = pd.DataFrame(output_data1)

    Out_df1.to_csv("output_roberta_modified.csv", index=False)


def main():
    wandb.init(project='roberta-parameter-search')

    config_defaults = {
        'max_seq_length': 512,
        'batch_size': 16,
        'learning_rate': 1e-5,
        'num_epochs': 10,
    }
    wandb.config.update(config_defaults)

    sweep_config = {
        'method': 'random',
        'parameters': {
            'max_seq_length': {'values': [128, 256, 512]},
            'batch_size': {'values': [8, 16, 32]},
            'learning_rate': {'values': [1e-4, 1e-5, 1e-6]},
            'num_epochs': {'values': [5, 10, 15]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project='roberta-parameter-search')

    def sweep_train():
        with wandb.init() as run:
            config = wandb.config
            train_model(config)

    wandb.agent(sweep_id, sweep_train)


if __name__ == "__main__":
    main()
