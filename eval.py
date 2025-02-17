import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re
import random

from main import LSTMClassifier

# ---------------------------
# Hyperparameters and settings
# ---------------------------
MAX_SEQ_LEN = 50  # Must match the training setting
BATCH_SIZE = 32

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Mapping label indices to emotion names
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
    """Lowercase and tokenize text into alphanumeric tokens."""
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens


def build_vocab(texts, min_freq=1):
    """Build vocabulary from texts."""
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab):
    """Convert text into a sequence of token indices."""
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    return sequence


def pad_sequence(seq, max_len):
    """Pad or truncate sequence to a fixed maximum length."""
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
# Main Evaluation Function
# ---------------------------
def main():
    # Load the dataset from Hugging Face
    dataset = load_dataset("dair-ai/emotion", "unsplit")
    texts = [item["text"] for item in dataset["train"]]
    labels = [item["label"] for item in dataset["train"]]

    # Split into train and test sets (using the same random_state as during training)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Rebuild the vocabulary from the training texts (should match the one used in training)
    vocab = build_vocab(train_texts, min_freq=1)

    # Create the test dataset and DataLoader
    test_dataset = EmotionDataset(test_texts, test_labels, vocab, max_len=MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the saved model (the entire model was saved)
    model_path = "lstm_emotion_model.pth"
    model = torch.load(model_path, weights_only=False)

    # Set device and put model in evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Evaluate overall test accuracy
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")

    # Show a few examples from the test set with detected emotions
    print("Some test examples with detected emotions:")
    num_examples = 5
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
