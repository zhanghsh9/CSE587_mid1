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
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import random

# ---------------------------
# Hyperparameters and settings
# ---------------------------
MAX_SEQ_LEN = 50  # Maximum tokens per sentence
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 300  # For word2vec-google-news-300 embeddings
HIDDEN_DIM = 128
NUM_CLASSES = 6  # 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
NUM_LAYERS = 2  # Number of LSTM layers
DROPOUT = 0.5  # Dropout probability for LSTM (applies to all layers except the last)

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
# Modified LSTM Classifier Model
# ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, pretrained_embeddings=None):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Optionally freeze the embeddings:
            # self.embedding.weight.requires_grad = False

        # Multi-layer LSTM. Note: dropout only applies if num_layers > 1.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Activation layer - you can change nn.ReLU() to nn.Tanh() if desired.
        self.activation = nn.ReLU()

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # shape: [batch_size, seq_len, embedding_dim]
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # When using multiple layers, h_n shape is [num_layers, batch_size, hidden_dim]
        # We use the hidden state from the last LSTM layer:
        hidden = h_n[-1]  # shape: [batch_size, hidden_dim]

        # Apply the activation function
        activated = self.activation(hidden)

        out = self.fc(activated)  # shape: [batch_size, num_classes]
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
    # Initialize embeddings randomly from a uniform distribution
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
# Main Training Loop with Plotting
# ---------------------------
def main():
    # Load the emotion dataset (unsplit) from Hugging Face
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

    # Optionally, load pretrained Word2Vec embeddings
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
                           num_layers=NUM_LAYERS, dropout=DROPOUT, pretrained_embeddings=pretrained_embeddings)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # For plotting loss curves
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on the test set for loss and accuracy
        model.eval()
        test_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        test_loss = test_running_loss / len(test_loader)
        test_losses.append(test_loss)
        accuracy = correct / total
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {accuracy * 100:.2f}%")

    # Save the model (entire model object)
    model_path = "lstm_emotion_model.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

    # ---------------------------
    # Plot and Save Training Curve (Loss vs Epoch)
    # ---------------------------
    epochs = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss vs Epoch")
    plt.legend()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()
    print("Loss curve saved to loss_curve.png")

    # ---------------------------
    # Compute ROC Curve on Test Set
    # ---------------------------
    # Gather predicted probabilities and true labels for the test set
    all_true = []
    all_scores = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            all_scores.extend(probs.cpu().numpy())
            all_true.extend(targets.cpu().numpy())

    all_true = np.array(all_true)
    all_scores = np.array(all_scores)

    # Binarize the true labels for ROC computation
    all_true_bin = label_binarize(all_true, classes=list(range(NUM_CLASSES)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(all_true_bin[:, i], all_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_true_bin.ravel(), all_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average ROC curve (area = {roc_auc['micro']:.2f})",
             color='deeppink', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue']
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"ROC curve of class {i} ({emotion_mapping[i]}) (area = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()
    print("ROC curve saved to roc_curve.png")


if __name__ == "__main__":
    main()
