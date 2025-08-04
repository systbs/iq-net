import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import LongformerModel, LongformerConfig
from datasets import load_dataset
import random
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import traceback

# ===========================================
# 0. Reproducibility Helper
# ===========================================
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===========================================
# 1. Text Data Handling & Preparation
# ===========================================
class TextVocabulary:
    def __init__(self, texts, max_size=20000):
        self.pad_token, self.unk_token = "<pad>", "<unk>"
        self.pad_idx, self.unk_idx = 0, 1
        word_counts = Counter(word for text in texts for word in text.split())
        self.vocab = [self.pad_token, self.unk_token] + [word for word, _ in word_counts.most_common(max_size - 2)]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}

    def text_to_sequence(self, text, max_len):
        seq = [self.word_to_idx.get(word, self.unk_idx) for word in text.split()]
        padded_seq = np.full(max_len, self.pad_idx, dtype=int)
        seq_len = min(len(seq), max_len)
        if seq_len > 0:
            padded_seq[:seq_len] = seq[:seq_len]
        return padded_seq

class IMDBTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        seq = self.vocab.text_to_sequence(self.texts[idx], self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def setup_text_data():
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", trust_remote_code=True)
    train_texts = [sample['text'] for sample in dataset['train']]
    train_labels = [sample['label'] for sample in dataset['train']]
    test_texts = [sample['text'] for sample in dataset['test']]
    test_labels = [sample['label'] for sample in dataset['test']]
    print("Building vocabulary...")
    vocab = TextVocabulary(train_texts + test_texts)
    max_len = max(max(len(text.split()) for text in train_texts), max(len(text.split()) for text in test_texts))
    return vocab, (train_texts, train_labels, test_texts, test_labels, max_len)

# ===========================================
# 2. Model Implementations
# ===========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.pe[:, :x.size(1), :]

class HolisticContextExtractor(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=-1)
        head_outputs = torch.sum(weights * v, dim=2)
        concatenated_heads = head_outputs.reshape(B, self.embed_dim)
        return self.combine(concatenated_heads)

class AssociativeContextExtractor(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.s_proj = nn.Linear(embed_dim, 1)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        context = torch.sum(weights * v, dim=1)
        return context

class ZarvanBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.holistic_ctx = HolisticContextExtractor(embed_dim, num_heads)
        self.associative_ctx = AssociativeContextExtractor(embed_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        self.update_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        q_holistic = self.holistic_ctx(x)
        q_associative = self.associative_ctx(x)
        q_holistic_exp = q_holistic.unsqueeze(1).expand(-1, S, -1)
        q_associative_exp = q_associative.unsqueeze(1).expand(-1, S, -1)
        gate_input = torch.cat([x, q_holistic_exp, q_associative_exp], dim=-1)
        input_gate, forget_gate = self.gate_net(gate_input).chunk(2, dim=-1)
        gated_x = torch.sigmoid(input_gate) * x + torch.sigmoid(forget_gate) * self.update_proj(x)
        output = self.ffn(gated_x)
        return self.norm(x + output)

class ZarvanModelText(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, num_classes: int, max_seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            ZarvanBlock(embed_dim, embed_dim * 2, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor, inputs_embeds=None) -> torch.Tensor:
        if inputs_embeds is None:
            x = self.embedding(x)
        else:
            x = inputs_embeds
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        y = self.output_head(x.mean(dim=1))
        return y

class LSTMModelText(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, inputs_embeds=None):
        h_initial = self.embedding(x) if inputs_embeds is None else inputs_embeds
        h_final, _ = self.lstm(h_initial)
        y = self.fc_out(h_final[:, -1, :])
        return y

class GRUModelText(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, inputs_embeds=None):
        h_initial = self.embedding(x) if inputs_embeds is None else inputs_embeds
        h_final, _ = self.gru(h_initial)
        y = self.fc_out(h_final[:, -1, :])
        return y

class TransformerModelText(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, num_layers, num_classes, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*2, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_classes)
    def forward(self, x, inputs_embeds=None):
        h_initial = self.embedding(x) if inputs_embeds is None else inputs_embeds
        h_final = self.encoder(self.pos_encoder(h_initial))
        y = self.fc_out(h_final.mean(dim=1))
        return y

class LongformerModelText(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, num_layers, num_classes, max_seq_len):
        super().__init__()
        config = LongformerConfig(vocab_size=vocab_size, hidden_size=embed_dim, num_attention_heads=n_heads, num_hidden_layers=num_layers, attention_window=64, pad_token_id=0, max_position_embeddings=max_seq_len)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.longformer = LongformerModel(config)
        self.fc_out = nn.Linear(embed_dim, num_classes)
    def forward(self, x, inputs_embeds=None):
        h_initial = self.embedding(x) if inputs_embeds is None else inputs_embeds
        attention_mask = torch.ones(h_initial.shape[:2], device=h_initial.device, dtype=torch.long)
        if x is not None: attention_mask = (x != 0).long()
        h_final = self.longformer(inputs_embeds=h_initial, attention_mask=attention_mask).last_hidden_state
        y = self.fc_out(h_final.mean(dim=1))
        return y

class CNNModelText(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in [3,4,5]])
        self.fc = nn.Linear(len([3,4,5]) * num_filters, num_classes)
        self.fc_out = self.fc
    def forward(self, x, inputs_embeds=None):
        h_initial = self.embedding(x) if inputs_embeds is None else inputs_embeds
        h = h_initial.permute(0, 2, 1)
        h_convs = [F.relu(conv(h)) for conv in self.convs]
        h_pooled = [F.max_pool1d(h_c, h_c.shape[2]).squeeze(2) for h_c in h_convs]
        h_final = torch.cat(h_pooled, dim=1)
        y = self.fc(h_final)
        return y

class TCNModelText(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.fc_out = nn.Linear(num_filters, num_classes)
    def forward(self, x, inputs_embeds=None):
        h_initial = self.embedding(x) if inputs_embeds is None else inputs_embeds
        h = h_initial.permute(0, 2, 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h_final = h.mean(dim=2)
        y = self.fc_out(h_final)
        return y

# ===========================================
# 3. Evaluation Function
# ===========================================
def evaluate_model(name, config, vocab, data, device):
    try:
        train_texts, train_labels, test_texts, test_labels, max_len = data
        train_dataset = IMDBTextDataset(train_texts, train_labels, vocab, max_len)
        test_dataset = IMDBTextDataset(test_texts, test_labels, vocab, max_len)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        model = create_text_model(name, config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"🚀 [Evaluating {name}] Started training on {device}...")
        
        # Training
        model.train()
        for epoch in range(config['max_epochs']):
            total_loss, count = 0, 0
            for x_batch, y_batch in train_loader:
                if x_batch.shape[0] < 2: continue
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = F.cross_entropy(predictions, y_batch)
                if torch.isnan(loss) or torch.isinf(loss): continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            print(f"  - Epoch {epoch+1}/{config['max_epochs']}, Loss: {total_loss/count:.4f}")
        
        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                if x_batch.shape[0] < 2: continue
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(predictions)
                all_labels.extend(y_batch.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"  - Evaluation finished. Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return {
            'Accuracy': accuracy,
            'F1_Score': f1,
            'Precision': precision,
            'Recall': recall
        }
    except Exception as e:
        print(f"❌ CRITICAL ERROR evaluating model {name}: {e}")
        traceback.print_exc()
        return None

# ===========================================
# 4. Plotting & Main Execution
# ===========================================
def plot_bar_chart(results, filename):
    model_names = list(results.keys())
    metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
    n_metrics = len(metrics)
    bar_width = 0.2
    index = np.arange(len(model_names))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0.0) for model in model_names]
        ax.bar(index + i * bar_width, values, bar_width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Standard Benchmark Results for All Models')
    ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 Starting Standard Text Benchmark...")
    vocab, data = setup_text_data()

    config = {
        'vocab_size': len(vocab.vocab), 
        'embed_dim': 128, 
        'hidden_dim': 256, 
        'num_layers': 2,
        'num_heads': 4, 
        'num_classes': 2, 
        'num_filters': 100, 
        'batch_size': 16,
        'max_epochs': 5,
        'max_seq_len': 4096,
    }

    model_names = ["CNN", "TCN", "LSTM", "GRU", "Transformer", "Longformer", "Zarvan"]
    results = {}

    for name in model_names:
        result = evaluate_model(name, config, vocab, data, device)
        if result:
            results[name] = result
            print(f"🎉 Successfully completed evaluation for {name}.\n")

    print("\n📊 Final Benchmark Results...")
    if not results:
        print("No models were successfully evaluated.")
        return

    # Save results to CSV
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.sort_values(by="Accuracy", ascending=False)
    print("\n" + "="*25 + " STANDARD BENCHMARK REPORT " + "="*25)
    print(df.to_markdown(floatfmt=".4f"))
    print("="*70 + "\n")
    df.to_csv('standard_benchmark_report.csv')
    
    # Plot bar chart
    plot_bar_chart(results, "standard_benchmark_bar_chart.png")
    print("  - Report and bar chart saved successfully.")

def create_text_model(name, config):
    model_map = {
        "CNN": CNNModelText, 
        "TCN": TCNModelText, 
        "LSTM": LSTMModelText, 
        "GRU": GRUModelText, 
        "Transformer": TransformerModelText, 
        "Longformer": LongformerModelText, 
        "Zarvan": ZarvanModelText
    }
    model_class = model_map[name]
    if name in ["Transformer", "Longformer", "Zarvan"]: 
        return model_class(
            config['vocab_size'], 
            config['embed_dim'], 
            config['num_heads'], 
            config['num_layers'], 
            config['num_classes'], 
            config['max_seq_len']
        )
    elif name in ["LSTM", "GRU"]: 
        return model_class(
            config['vocab_size'], 
            config['embed_dim'], 
            config['hidden_dim'], 
            config['num_layers'], 
            config['num_classes']
        )
    else: 
        return model_class(
            config['vocab_size'], 
            config['embed_dim'], 
            config['num_filters'], 
            config['num_classes']
        )

if __name__ == '__main__':
    main()