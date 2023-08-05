import pandas as pd
import numpy as np
import nltk as nltk
import pickle
import re
import sklearn
from nltk import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


train_data_path = "data/train.jsonl"
valid_data_path = "data/val.jsonl"
test_data_path = "data/test.jsonl"

df = pd.read_json(train_data_path, lines=True)
df2 = pd.read_json(valid_data_path, lines = True)
df3 = pd.read_json(test_data_path, lines = True)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]

def preprocess_data(df):
    #df = df[['postText','targetParagraphs', 'targetTitle']]
    df = df[['postText','targetParagraphs']]

    # convert all columns into strings
    #df.loc[:, ['postText', 'targetParagraphs', 'targetTitle']] = df.loc[:, ['postText', 'targetParagraphs', 'targetTitle']].astype(str)
    df.loc[:, ['postText', 'targetParagraphs']] = df.loc[:, ['postText', 'targetParagraphs']].astype(str)

    #tokenize the relevant columns (not actually used for the Bag of Word approach)
    tokenizer = RegexpTokenizer(r"\w+")
    df["postText_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["postText"]), axis = 1)
    df["paragraph_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetParagraphs"]), axis = 1)
    #df["targetTitle_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetTitle"]), axis = 1)
    
    #removing stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    df["postText_tokens"] = df.apply(lambda row: [element for element in row["postText_tokens"] if element not in stopwords], axis = 1)
    df["paragraph_tokens"] = df.apply(lambda row: [element for element in row["paragraph_tokens"] if element not in stopwords], axis = 1)
    #df["targetTitle_tokens"] = df.apply(lambda row: [element for element in row["targetTitle_tokens"] if element not in stopwords], axis = 1)
    
    #lowercasing 
    df['postText_tokens'] = df['postText_tokens'].map(lambda row: list(map(str.lower, row)))
    df['paragraph_tokens'] = df['paragraph_tokens'].map(lambda row: list(map(str.lower, row)))
   # df['targetTitle_tokens'] = df['targetTitle_tokens'].map(lambda row: list(map(str.lower, row)))
    
    # multiple space to single space and remove special characters
   # df[['postText_tokens', 'paragraph_tokens', 'targetTitle_tokens']] = df[['postText_tokens', 'paragraph_tokens', 'targetTitle_tokens']].replace(r'\s+', ' ', regex=True).replace(r'\W', ' ', regex = True)
    
    df[['postText_tokens', 'paragraph_tokens']] = df[['postText_tokens', 'paragraph_tokens']].replace(r'\s+', ' ', regex=True).replace(r'\W', ' ', regex = True)
    
    #lemmatize tokens
    df['postText_tokens'] = df['postText_tokens'].apply(lemmatize_text)
    df['paragraph_tokens'] = df['paragraph_tokens'].apply(lemmatize_text)
   # df['targetTitle_tokens'] = df['targetTitle_tokens'].apply(lemmatize_text)
  
   #df['combined_texts'] = df['postText_tokens'].apply(lambda tokens: ' '.join(tokens)) + " " + df['paragraph_tokens'].apply(lambda tokens: ' '.join(tokens)) + " " + df['targetTitle_tokens'].apply(lambda tokens: ' '.join(tokens))
    df['combined_texts'] = df['postText_tokens'].apply(lambda tokens: ' '.join(tokens)) + " " + df['paragraph_tokens'].apply(lambda tokens: ' '.join(tokens))

    return df

def vector():
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    return vectorizer

def tuneModel(vectorizer, X_train, X_val, Y_train, Y_val):
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
    maxi = 0
    hyper = 0
    
    best_model_state = None
    
    for i in tqdm(alpha):
        classifier = MultinomialNB(alpha=i)
        t_pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
        t_pipeline.fit(X_train, Y_train)
        y_pred = t_pipeline.predict(X_val)
        score = accuracy_score(Y_val, y_pred)
        if score > maxi:
            maxi = score
            hyper = i
            filename = f"model_mnb.pkl"
            with open(filename, 'wb') as file:
                pickle.dump(t_pipeline, file)
        print(score)
    return t_pipeline, hyper


#def trainModel(vectorizer, hyper, X_train, X_val, Y_train, Y_val):
 #   classifier = MultinomialNB(alpha=hyper)
  #  pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
   # pipeline.fit(X_train, Y_train)
    #with open('model_mnb.pkl', 'wb') as file:
     #   pickle.dump(pipeline, file)
    #return pipeline

# Function to test the model
#def testModel(pipeline, hyper, X_test, Y_test):
 #   y_pred = pipeline.predict(X_test)
  #  score = accuracy_score(Y_test, y_pred)
   # print(score)

# Function  
def main():    
    train_selected = preprocess_data(df)
    val_selected = preprocess_data(df2)
    test_selected = preprocess_data(df3)
    
    # Create an instance of LabelEncoder
    label_encoder = LabelEncoder()

    # Flatten the multi-labels into a single list for each sample
    train_labels = [tag for tags in df['tags'] for tag in tags]
    #val_labels = [tag for tags in df2['tags'] for tag in tags]

    # Fit the label encoder on the flattened labels
    label_encoder.fit(train_labels)

    # Transform the multi-labels to integer labels
    train_y = [label_encoder.transform(tags) for tags in df['tags']]
    val_y = [label_encoder.transform(tags) for tags in df2['tags']]

    X_train = train_selected['combined_texts']
    X_val = val_selected['combined_texts']
    X_test = test_selected['combined_texts']

    vectorizer = vector()
    pipeline, hyper = tuneModel(vectorizer, X_train, X_val, train_y, val_y)
    #pipeline = trainModel(vectorizer, hyper, X_train, X_val, train_y, val_y)
    test_pred = pipeline.predict(X_test)
    test_pred_flattened = test_pred.ravel()
    test_pred_tags = label_encoder.inverse_transform(test_pred_flattened)
    output_data = []
    for index, pred in enumerate(test_pred_tags):
        prediction = {'id': index, 'spoilerType': pred}
        output_data.append(prediction)
    Out_df = pd.DataFrame(output_data)
    Out_df.to_csv("output_mnb_uni_bi.csv", index=False)

if __name__ == "__main__":
    main()