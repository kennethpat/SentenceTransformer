import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer(nn.Module):
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased", 
                 embedding_dim: int = 768, 
                 pooling_strategy: str = "mean", 
                 max_length: int = 128) -> None:
        super(SentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length

        self.use_projection = None
        if embedding_dim != self.transformer.config.hidden_size:
            self.use_projection = True
        else:
            self.use_projection = False

        if self.use_projection is True:
            self.projection = nn.Linear(self.transformer.config.hidden_size, embedding_dim)

    def forward(self, sentences: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        device = next(self.transformer.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
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

    def encode(self, sentences: list[str], batch_size: int = 32, normalize: bool = False) -> np.ndarray:
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                embeddings = self.forward(batch)
                if normalize is True:
                    embeddings = nn.functional.normalize(embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings
    
def main() -> None:
    test_model = SentenceTransformer(model_name="distilbert-base-uncased", embedding_dim=384, pooling_strategy="mean", max_length=128)
    test_sentences = [
            "This is a simple sentence to encode.",
            "The quick brown fox jumps over the lazy dog.",
            "Sentence transformers are useful for many NLP tasks.",
            "This sentence is similar to the first one but with different words.",
            "Machine learning models can process natural language effectively."
    ]
    test_embeddings = test_model.encode(sentences=test_sentences, normalize=True)
    print(f"Generated {test_embeddings.shape[0]} embeddings with dimension {test_embeddings.shape[1]}")
    print(f"Sample of the first embedding vector:\n{test_embeddings[0][:10]}")
    print("Computing similarities between sentences:")
    for i in range(len(test_sentences)):
        for j in range(i+1, len(test_sentences)):
            similarity = np.dot(test_embeddings[i], test_embeddings[j])
            print(f"Similarity between sentence {i+1} and {j+1}: {similarity:.4f}")

if __name__ == "__main__":
    main()