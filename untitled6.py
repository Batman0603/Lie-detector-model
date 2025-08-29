# -*- coding: utf-8 -*-
import os
import subprocess
import zipfile
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import numpy as np

# Make sure you have your kaggle.json placed at:
# C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
# (no need to write code for this in Windows / VS Code)

# -----------------------------
# 1. Download and unzip datasets
# -----------------------------
def download_dataset(dataset_name):
    print(f"Downloading {dataset_name} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name])

def unzip_file(zip_file, extract_to):
    if os.path.exists(zip_file):
        print(f"Unzipping {zip_file} to {extract_to} ...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"{zip_file} not found!")

datasets = [
    ("csmalarkodi/liar-fake-news-dataset", "liar-fake-news-dataset.zip", "liar-fake-news-dataset"),
    ("clmentbisaillon/fake-and-real-news-dataset", "fake-and-real-news-dataset.zip", "fake-and-real-news-dataset"),
    ("abhinavwalia95/entity-annotated-corpus", "entity-annotated-corpus.zip", "entity-annotated-corpus"),
    ("mrisdal/fake-news", "fake-news.zip", "fake-news"),
    ("arashnic/fake-claim-dataset", "fake-claim-dataset.zip", "fake-claim-dataset")
]

for kaggle_name, zip_file, folder in datasets:
    download_dataset(kaggle_name)
    unzip_file(zip_file, folder)

# -----------------------------
# 2. Load and process data
# -----------------------------
# LIAR Dataset
df1 = pd.read_csv("liar-fake-news-dataset/train.tsv", sep='\t', header=None)
df1 = df1.rename(columns={2: 'text', 1: 'label'})
lie_labels = ['pants-fire', 'false', 'barely-true']
df1['label'] = df1['label'].apply(lambda x: 0 if x in lie_labels else 1)

# Fake & Real News
df_fake = pd.read_csv("fake-and-real-news-dataset/Fake.csv")
df_real = pd.read_csv("fake-and-real-news-dataset/True.csv")
df_fake['label'] = 0
df_real['label'] = 1
df2 = pd.concat([df_fake, df_real])

# Entity Annotated Corpus
df3 = pd.read_csv("entity-annotated-corpus/ner.csv", encoding='ISO-8859-1', engine='python', sep=',', quotechar='"', on_bad_lines='skip')
df3['tag'] = df3['tag'].apply(lambda x: 1 if x == 'O' else 0)
df3 = df3.rename(columns={'word': 'text', 'tag': 'label'})

# Political Fake News Dataset
df4 = pd.read_csv("fake-news/fake.csv")
df4 = df4.rename(columns={'type': 'label'})
fake_types = ['fake', 'bs']
df4['label'] = df4['label'].apply(lambda x: 0 if x in fake_types else 1)

# Fake Claim Dataset
df5 = pd.read_csv("fake-claim-dataset/train.csv")
df5 = df5.rename(columns={'statement': 'text'})
lie_labels = ['pants-fire', 'false', 'barely-true']
df5['label'] = df5['label'].apply(lambda x: 0 if x in lie_labels else 1)

# -----------------------------
# 3. Combine all datasets
# -----------------------------
all_dfs = [
    df1[['text', 'label']],
    df2[['text', 'label']],
    df3[['text', 'label']],
    df4[['text', 'label']],
    df5[['text', 'label']]
]

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.dropna(subset=['text', 'label'], inplace=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Combined Dataset Ready!")
print("Total samples:", combined_df.shape[0])
print(combined_df['label'].value_counts())

# -----------------------------
# 4. Preprocess text
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

combined_df['cleaned'] = combined_df['text'].astype(str).apply(preprocess)




# -----------------------------
# 5. BERT embeddings
# -----------------------------

from sentence_transformers import SentenceTransformer
import pandas as pd # Added import for pandas

# Load a lightweight, fast S-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert to sentence embeddings
# Ensure combined_df and 'cleaned' column exist by running previous cells
if 'combined_df' not in locals() or 'cleaned' not in combined_df.columns:
    print("Error: 'combined_df' or 'cleaned' column not found. Please run the preceding cells.")
else:
    X = model.encode(combined_df['cleaned'], show_progress_bar=True)

    # Labels
    y = combined_df['label'].astype(int).values

    print("✅ BERT Embedding Shape:", len(X), "x", len(X[0]))



#======================================
# Train test split
#======================================

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load the BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Generate dense vector embeddings for each sentence
X_bert = bert_model.encode(combined_df['cleaned'], show_progress_bar=True)

# 3. Get labels
y = combined_df['label'].astype(int).values

# 4. Train-test split (Stratified to preserve label balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_bert, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Convert to numpy arrays if needed (some ML models require this)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# ✅ Output the shapes
print("✅ BERT Embedding Shape:", X_train.shape[1], "features")
print("Train set:", X_train.shape, "| Test set:", X_test.shape)


#======================================
# k fold cv
#======================================

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Train model (using RandomForest as example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    # Predict on validation fold
    y_pred = model.predict(X_val)

    # Compute metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Store results
    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)

    print(f"Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# ✅ Mean scores across all folds
print("\nAverage metrics across folds:")
print(f"Accuracy: {np.mean(accuracy_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall: {np.mean(recall_list):.4f}")
print(f"F1 Score: {np.mean(f1_list):.4f}")
#==========================================
# -----------------------------
# 6. Train a classifier (Logistic Regression)
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Predict on user input
# -----------------------------
while True:
    user_text = input("\nEnter a news statement (or type 'exit' to quit): ")
    if user_text.lower() == 'exit':
        break

    # Preprocess like before
    user_clean = preprocess(user_text)

    # Embed
    user_embed = model.encode([user_clean])

    # Predict
    user_pred = clf.predict(user_embed)[0]
    user_prob = clf.predict_proba(user_embed)[0]

    print(f"Prediction: {'True' if user_pred==1 else 'Lie'} (confidence: {user_prob[user_pred]:.2f})")
