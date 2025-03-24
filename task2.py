import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import Any, Optional

class SentenceTransformerMultiTask(nn.Module):
    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 embedding_dim: int = 768,
                 pooling_strategy: str = "mean",
                 max_length: int = 128,
                 num_sentiment_classes: int = 3) -> None:
        super(SentenceTransformerMultiTask, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.transformer_dim = self.transformer.config.hidden_size

        self.use_projection = None
        if embedding_dim != self.transformer_dim:
            self.use_projection = True
        else:
            self.use_projection = False

        if self.use_projection is True:
            self.projection = nn.Linear(self.transformer_dim, embedding_dim)
            self.shared_dim = embedding_dim
        else:
            self.shared_dim = self.transformer_dim

        self.sentiment_classifier = nn.Sequential(nn.Linear(self.shared_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_sentiment_classes))
        self.sentiment_labels = ["negative", "neutral", "positive"]

    def _extract_features(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        device = next(self.transformer.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.transformer(**inputs)
        hidden_states = outputs.last_hidden_state
        
        if self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0]
        elif self.pooling_strategy == "max":
            pooled = torch.max(hidden_states, dim=1)[0]
        else:
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        if self.use_projection is True:
            pooled = self.projection(pooled)

        return pooled

    def forward(self, sentences: list[str], task: Optional[str] = None) -> dict[str, torch.Tensor]:
        shared_features = self._extract_features(sentences)
        outputs = {}
        
        if task is None or task == "embedding":
            outputs["embedding"] = shared_features

        if task is None or task == "sentiment":
            sentiment_logits = self.sentiment_classifier(shared_features)
            outputs["sentiment"] = sentiment_logits

        return outputs

    def encode(self, sentences: list[str], batch_size: int = 32, normalize: bool = False) -> np.ndarray:
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                outputs = self.forward(batch, task="embedding")
                embeddings = outputs["embedding"]
                if normalize is True:
                    embeddings = nn.functional.normalize(embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def predict_sentiment(self, sentences: list[str], batch_size: int = 32) -> dict[str, list[str]]:
        self.eval()
        all_predictions = []
        all_probabilities = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                outputs = self.forward(batch, task="sentiment")
                logits = outputs["sentiment"]
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        prediction_labels = [self.sentiment_labels[pred] for pred in all_predictions]
        all_probabilities = np.vstack(all_probabilities)
        results = {"labels": prediction_labels, "probabilities": all_probabilities}
        
        return results
    
class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, str]:
        text = self.texts[idx]
        label = self.labels[idx]
        item = {"text": text, "label": label}
        return item
    
def train_multitask_model(
    model: SentenceTransformerMultiTask, 
    train_texts: list[str], 
    train_labels: list[int], 
    val_texts: Optional[list[str]] = None, 
    val_labels: Optional[list[int]] = None, 
    epochs: int = 5, 
    batch_size: int = 16, 
    learning_rate: float = 2e-5) -> tuple[SentenceTransformerMultiTask, dict[str, list[float]]]:

    assert isinstance(epochs, int), "'epochs' must be an integer."
    assert epochs > 0, "'epochs' must be positive."

    train_dataset = SentimentDataset(train_texts, train_labels, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, model.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    device = next(model.parameters()).device
    model.train()

    training_stats = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": []
    }

    for epoch in range(epochs):
        print(f"===== Epoch {epoch + 1} / {epochs} =====")
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            texts = batch["text"]
            labels = batch["label"]
            labels = labels.to(device)
            outputs = model(texts, task="sentiment")
            logits = outputs["sentiment"]
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_stats["train_losses"].append(avg_train_loss)

        print(f"Train Loss: {avg_train_loss:.4f}")

        if val_loader is True:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    texts = batch["text"]
                    labels = batch["label"]
                    labels = labels.to(device)

                    outputs = model(texts, task="sentiment")
                    logits = outputs["sentiment"]
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            training_stats["val_losses"].append(avg_val_loss)
            training_stats["val_accuracies"].append(val_accuracy)

            print(f"Validation Loss: {avg_val_loss:.4f}\nAccuracy: {val_accuracy:.4f}")

    return model, training_stats

def main() -> None:
    test_model = SentenceTransformerMultiTask(model_name="distilbert-base-uncased", embedding_dim=384, pooling_strategy="mean", max_length=128, num_sentiment_classes=3)
    test_sentences = [
        "This is a simple sentence to encode.",
        "The quick brown fox jumps over the lazy dog.",
        "I absolutely love this product, it's amazing!",
        "This movie was terrible and a complete waste of time.",
        "The service was okay, nothing special but not bad either."
    ]

    print("GENERATING SENTENCE EMBEDDINGS:")
    test_embeddings = test_model.encode(sentences=test_sentences, normalize=True)
    print(f"Generated {test_embeddings.shape[0]} embeddings with dimension {test_embeddings.shape[1]}")
    print(f"Sample of the first embedding vector:\n{test_embeddings[0][:10]}")
    print("PREDICTING SENTIMENT:")
    test_sentiment_results = test_model.predict_sentiment(test_sentences)

    for i, (text, label) in enumerate(zip(test_sentences, test_sentiment_results["labels"])):
        probs = test_sentiment_results["probabilities"][i]
        print(f"Text: {text}")
        print(f"Predicted sentiment: {label}")
        print(f"Class probabilities: Negative = {probs[0]:.4f}, Neutral = {probs[1]:.4f}, Positive = {probs[2]:.4f}\n")

    print("TRAINING EXAMPLE (with dummy data):")
    test_train_texts = [
        "I love this product.",
        "This is terrible.",
        "It's okay I guess.",
        "Best purchase ever!",
        "Complete waste of money.",
        "It works as expected."
    ]
    test_train_labels = [2, 0, 1, 2, 0, 1]
    print("Training the model on sentiment analysis task...")
    test_model, test_stats = train_multitask_model(
        model=test_model, 
        train_texts=test_train_texts, 
        train_labels=test_train_labels, 
        epochs=20, 
        batch_size=2)

    print("Re-testing sentiment prediction after training:")
    test_sentiment_results = test_model.predict_sentiment(test_sentences)
    for i, (text, label) in enumerate(zip(test_sentences, test_sentiment_results["labels"])):
        probs = test_sentiment_results["probabilities"][i]
        print(f"Text: {text}")
        print(f"Predicted sentiment: {label}")
        print(f"Class probabilities: Negative = {probs[0]:.4f}, Neutral = {probs[1]:.4f}, Positive = {probs[2]:.4f}\n")

if __name__ == "__main__":
    main()
