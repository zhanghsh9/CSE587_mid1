# Code for CSE587 mid term 1
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import random

# ---------------------------
# Hyperparameters and settings
# ---------------------------
MAX_SEQ_LEN = 50  # Maximum number of tokens per sentence
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 300  # For word2vec-google-news-300 embeddings
HIDDEN_DIM = 128
NUM_CLASSES = 6  # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise

# Special tokens for padding and unknown words:
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Mapping label indices to emotion names:
emotion_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


# ---------------------------
# Utility functions
# ---------------------------
def tokenize(text):
    """Simple tokenizer: lowercase and extract alphanumeric words."""
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


def build_vocab(texts, min_freq=1):
    """Build a vocabulary dictionary mapping word to index."""
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    # Reserve indices for PAD and UNK tokens
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab):
    """Convert text into a list of token indices based on the vocabulary."""
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    return sequence


def pad_sequence(seq, max_len):
    """Pad or truncate a sequence to a fixed maximum length."""
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


# ---------------------------
# PyTorch Dataset
# ---------------------------
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        seq = text_to_sequence(text, self.vocab)
        seq = pad_sequence(seq, self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ---------------------------
# LSTM Classifier Model
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_embeddings=None):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Optionally freeze the embedding weights:
            # self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # shape: [batch_size, seq_len, embedding_dim]
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # h_n shape: [1, batch_size, hidden_dim]
        hidden = h_n.squeeze(0)  # shape: [batch_size, hidden_dim]
        out = self.fc(hidden)  # shape: [batch_size, num_classes]
        return out


# ---------------------------
# Load Pretrained Word2Vec Embeddings
# ---------------------------
def load_pretrained_embeddings(vocab, embedding_dim):
    """
    Load pretrained embeddings from gensim.
    Here we use 'word2vec-google-news-300' which can be downloaded via gensim's API.
    (Note: This model is large (~1.6GB) and may take some time to download.)
    """
    print("Loading pretrained word embeddings...")
    try:
        wv = api.load("word2vec-google-news-300")
    except Exception as e:
        print("Error loading pretrained embeddings:", e)
        return None

    vocab_size = len(vocab)
    # Initialize with a uniform distribution
    embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embeddings[vocab[PAD_TOKEN]] = np.zeros(embedding_dim)  # zero vector for padding
    found = 0
    for word, idx in vocab.items():
        if word in wv:
            embeddings[idx] = wv[word]
            found += 1
    print(f"Found pretrained embeddings for {found}/{vocab_size} words.")
    return torch.tensor(embeddings, dtype=torch.float)


# ---------------------------
# Main Training Loop
# ---------------------------
def main():
    # Load the emotion dataset (unsplit) from Hugging Face datasets
    dataset = load_dataset("dair-ai/emotion", "unsplit")
    texts = [item["text"] for item in dataset["train"]]
    labels = [item["label"] for item in dataset["train"]]

    # Split into train and test sets (80/20 split)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Build vocabulary from the training texts
    vocab = build_vocab(train_texts, min_freq=1)
    print("Vocabulary size:", len(vocab))

    # Optionally, load pretrained Word2Vec embeddings (set use_pretrained = True to use them)
    use_pretrained = True
    if use_pretrained:
        pretrained_embeddings = load_pretrained_embeddings(vocab, EMBEDDING_DIM)
    else:
        pretrained_embeddings = None

    # Create PyTorch datasets and dataloaders
    train_dataset = EmotionDataset(train_texts, train_labels, vocab, max_len=MAX_SEQ_LEN)
    test_dataset = EmotionDataset(test_texts, test_labels, vocab, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES,
                           pretrained_embeddings=pretrained_embeddings)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Evaluation on the test set during training
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    # Save the model after training
    model_path = "lstm_emotion_model.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

    # Show some examples from the test set with detected emotions
    print("\nSome test examples with detected emotions:")
    model.eval()
    num_examples = 5  # number of examples to display
    # Randomly choose examples from the test set:
    indices = random.sample(range(len(test_texts)), num_examples)

    for idx in indices:
        text = test_texts[idx]
        true_label = test_labels[idx]
        seq = text_to_sequence(text, vocab)
        seq = pad_sequence(seq, MAX_SEQ_LEN)
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(seq_tensor)
            predicted_label = torch.argmax(outputs, dim=1).item()

        print(f"Text: {text}")
        print(f"Ground Truth: {emotion_mapping[true_label]}, Predicted: {emotion_mapping[predicted_label]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
