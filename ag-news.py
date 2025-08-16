# ============================================================================
#
#   AG News Benchmark for IQ-NET Model Validation
#
#   This script benchmarks the five models from the IQ-NET paper
#   (Zarvan, LSTM, GRU, Transformer, CNN_1D) on the standard AG News
#   4-class topic classification task.
#
#   - It uses the Hugging Face `datasets` library to load ag_news.
#   - It builds a vocabulary from the training data.
#   - It trains each model and evaluates its performance.
#   - It reports multiclass classification metrics (Accuracy, Macro F1, etc.).
#
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import math
import random

# ===========================================
# 0. Reproducibility & Configuration
# ===========================================
def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- BENCHMARK CONFIGURATION ---
CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 4,
    'max_length': 96,      # Increased max length for news articles
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# --- MODEL CONFIGURATIONS (adapted from benchmark.py) ---
MODEL_CONFIGS = {
    'Zarvan': {'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.1},
    'LSTM': {'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2},
    'GRU': {'embed_dim': 64, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2},
    'Transformer': {'embed_dim': 64, 'num_heads': 4, 'num_layers': 2, 'ffn_dim': 128, 'dropout': 0.1},
    'CNN_1D': {'embed_dim': 64, 'num_channels': [64, 128], 'kernel_sizes': [3, 3], 'dropout': 0.2},
}

# ===========================================
# 1. Model Implementations (Copied from benchmark.py)
# ===========================================

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class _HolisticExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = torch.nn.functional.softmax(s, dim=2)
        head_outputs = torch.sum(weights * v, dim=2, keepdim=True)
        return self.combine(head_outputs.reshape(B, 1, E))

class _AssociativeExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj, self.v_proj = nn.Linear(embed_dim, 1), nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        s, v = self.s_proj(x), self.v_proj(x)
        return torch.sum(torch.nn.functional.softmax(s, dim=1) * v, dim=1, keepdim=True)

class _SequentialExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.v_proj, self.norm = nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)
        self.angle_calculator = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
    def forward(self, x):
        B, S, E = x.shape
        weights = torch.cumsum(self.s_proj(x) * self.v_proj(x), dim=1)
        omega = self.norm(self.angle_calculator(weights / S)) * math.pi
        phases = torch.cat([torch.cos(omega), torch.sin(omega)], dim=-1)
        return self.out_proj(phases)

class _ZarvanBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.input_adapter = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.holistic_ctx, self.associative_ctx, self.sequential_ctx = _HolisticExtractor(embed_dim, num_heads), _AssociativeExtractor(embed_dim), _SequentialExtractor(embed_dim)
        self.expert_gate = nn.Sequential(nn.Linear(embed_dim, 3), nn.SiLU())
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x_adapted = self.input_adapter(x)
        q_h, q_a, q_s = self.holistic_ctx(x_adapted), self.associative_ctx(x_adapted), self.sequential_ctx(x_adapted)
        g_h, g_a, g_s = self.expert_gate(x_adapted).chunk(3, dim=-1)
        h_candidate = (g_h * q_h.expand_as(x) + g_a * q_a.expand_as(x) + g_s * q_s)
        return x + self.ffn(self.norm(h_candidate))

class ZarvanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        self.layers = nn.ModuleList([_ZarvanBlock(config['embed_dim'], config['hidden_dim'], config['num_heads']) for _ in range(config['num_layers'])])
        self.output_head = nn.Linear(config['embed_dim'], config['num_classes'])
    def forward(self, x):
        h = self.pos_encoder(self.embedding(x))
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h.mean(dim=1))

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.lstm = nn.LSTM(config['embed_dim'], config['hidden_dim'], config['num_layers'], batch_first=True, dropout=config['dropout'])
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])
    def forward(self, x):
        h = self.embedding(x)
        _, (hidden, _) = self.lstm(h)
        return self.output_head(hidden[-1])

class GRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.gru = nn.GRU(config['embed_dim'], config['hidden_dim'], config['num_layers'], batch_first=True, dropout=config['dropout'])
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])
    def forward(self, x):
        h = self.embedding(x)
        _, hidden = self.gru(h)
        return self.output_head(hidden[-1])

class SimpleTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=config['num_heads'], dim_feedforward=config['ffn_dim'], dropout=config['dropout'], activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        self.output_head = nn.Linear(config['embed_dim'], config['num_classes'])
    def forward(self, x):
        h = self.pos_encoder(self.embedding(x))
        features = self.transformer_encoder(h)
        return self.output_head(features.mean(dim=1))

class CNN1DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        layers = []
        in_channels = config['embed_dim']
        for out_channels, kernel_size in zip(config['num_channels'], config['kernel_sizes']):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(config['num_channels'][-1], config['num_classes'])
    def forward(self, x):
        h = self.embedding(x).transpose(1, 2)
        conv_out = self.conv_layers(h)
        pooled_out = torch.max(conv_out, dim=2)[0]
        return self.output_head(pooled_out)

# ===========================================
# 2. Data Loading & Preprocessing
# ===========================================

class Vocabulary:
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.word2idx = {word: i for i, word in enumerate(specials)}
        self.word2idx.update({word: i + len(specials) for i, (word, _) in enumerate(counter.items())})
    def __len__(self):
        return len(self.word2idx)

def build_vocab(texts):
    counter = Counter(word for text in texts for word in text.split())
    return Vocabulary(counter)

def preprocess_data(dataset_split, vocab, max_length):
    all_indices, all_labels = [], []
    # AG News uses 'text' field instead of 'sentence'
    for item in dataset_split:
        tokens = item['text'].lower().split()
        indices = [vocab.word2idx.get(token, vocab.word2idx['<unk>']) for token in tokens]
        
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices += [vocab.word2idx['<pad>']] * (max_length - len(indices))
            
        all_indices.append(indices)
        all_labels.append(item['label'])
        
    return torch.LongTensor(all_indices), torch.LongTensor(all_labels)

# ===========================================
# 3. Training and Evaluation Loop
# ===========================================

def train_and_evaluate(model, train_loader, test_loader, config):
    model.to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Training]")
        for sentences, labels in train_loop:
            sentences, labels = sentences.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #train_loop.set_postfix(loss=loss.item())

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sentences, labels in test_loader:
                sentences, labels = sentences.to(config['device']), labels.to(config['device'])
                outputs = model(sentences)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Use 'macro' average for multiclass metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch} | Test Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}\n")
        
        if f1 > best_f1:
            best_f1 = f1
            final_metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}
            
    return final_metrics

# ===========================================
# 4. Main Execution
# ===========================================

if __name__ == '__main__':
    set_seed(42)
    
    print("--- 1. Loading and Preprocessing AG News Dataset ---")
    ag_news_dataset = load_dataset("ag_news")
    # AG News has a standard train/test split, no validation
    train_texts = [item['text'] for item in ag_news_dataset['train']]
    vocab = build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    train_data, train_labels = preprocess_data(ag_news_dataset['train'], vocab, CONFIG['max_length'])
    test_data, test_labels = preprocess_data(ag_news_dataset['test'], vocab, CONFIG['max_length'])
    
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=CONFIG['batch_size'], shuffle=False)
    print("Dataset preparation complete.\n")
    
    results = {}
    models_to_test = {
        "Zarvan": ZarvanModel,
        "Transformer": SimpleTransformerModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "CNN_1D": CNN1DModel,
    }
    
    for name, model_class in models_to_test.items():
        print(f"--- 2. Benchmarking {name} ---")
        
        model_config = MODEL_CONFIGS[name]
        model_config['vocab_size'] = len(vocab)
        model_config['num_classes'] = 4 # AG News has 4 classes
        
        model = model_class(model_config)
        
        start_time = time.time()
        metrics = train_and_evaluate(model, train_loader, test_loader, CONFIG)
        end_time = time.time()
        
        metrics['Training Time (s)'] = end_time - start_time
        results[name] = metrics

    print("\n" + "="*50)
    print("      AG News Benchmark Final Results")
    print("="*50)
    
    results_df = pd.DataFrame(results).T
    print(results_df.to_string(float_format="%.4f"))
    print("="*50)