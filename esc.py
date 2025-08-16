# ============================================================================
#
#   Speech Commands Benchmark for IQ-NET Model Validation
#
#   This script benchmarks the five models on the Google Speech Commands
#   dataset. It tests the models' ability to classify spoken words.
#
#   - It uses `torchaudio` to load the dataset and process audio waveforms.
#   - Audio waveforms are converted to Mel Spectrograms.
#   - The spectrogram is treated as a sequence of time frames, which is
#     fed into the sequence models.
#
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import time
import math
import random
import os # <-- Added import

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
    'batch_size': 64,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'sample_rate': 16000,
    'n_mels': 64,
    'n_fft': 1024,
    'hop_length': 512,
}

# --- MODEL CONFIGURATIONS ---
MODEL_CONFIGS = {
    'Zarvan': {'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.2},
    'LSTM': {'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.3},
    'GRU': {'embed_dim': 128, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.3},
    'Transformer': {'embed_dim': 128, 'num_heads': 4, 'num_layers': 2, 'ffn_dim': 256, 'dropout': 0.2},
    'CNN_1D': {'embed_dim': 128, 'num_channels': [64, 128], 'kernel_sizes': [3, 3], 'dropout': 0.3},
}

# ===========================================
# 1. Model Implementations (Identical to Vision tasks)
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
        self.s_proj, self.v_proj, self.combine = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s, v = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3), v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        head_outputs = torch.sum(torch.nn.functional.softmax(s, dim=2) * v, dim=2, keepdim=True)
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
        self.v_proj, self.norm, self.angle_calculator, self.out_proj = nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim * 2, embed_dim)
    def forward(self, x):
        B, S, E = x.shape
        weights = torch.cumsum(self.s_proj(x) * self.v_proj(x), dim=1)
        omega = self.norm(self.angle_calculator(weights / S)) * math.pi
        return self.out_proj(torch.cat([torch.cos(omega), torch.sin(omega)], dim=-1))

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

class AudioZarvanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config['input_dim'], config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        self.layers = nn.ModuleList([_ZarvanBlock(config['embed_dim'], config['hidden_dim'], config['num_heads']) for _ in range(config['num_layers'])])
        self.output_head = nn.Linear(config['embed_dim'], config['num_classes'])
    def forward(self, x):
        h = self.pos_encoder(self.input_proj(x))
        for layer in self.layers: h = layer(h)
        return self.output_head(h.mean(dim=1))

class AudioLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config['input_dim'], config['embed_dim'])
        self.lstm = nn.LSTM(config['embed_dim'], config['hidden_dim'], config['num_layers'], batch_first=True, dropout=config['dropout'])
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])
    def forward(self, x):
        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)
        return self.output_head(lstm_out.mean(dim=1))

class AudioGRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config['input_dim'], config['embed_dim'])
        self.gru = nn.GRU(config['embed_dim'], config['hidden_dim'], config['num_layers'], batch_first=True, dropout=config['dropout'])
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])
    def forward(self, x):
        h = self.input_proj(x)
        gru_out, _ = self.gru(h)
        return self.output_head(gru_out.mean(dim=1))

class AudioSimpleTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config['input_dim'], config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=config['num_heads'], dim_feedforward=config['ffn_dim'], dropout=config['dropout'], activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        self.output_head = nn.Linear(config['embed_dim'], config['num_classes'])
    def forward(self, x):
        h = self.pos_encoder(self.input_proj(x))
        features = self.transformer_encoder(h)
        return self.output_head(features.mean(dim=1))

class AudioCNN1DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config['input_dim'], config['embed_dim'])
        layers = []
        in_channels = config['embed_dim']
        for out_channels, kernel_size in zip(config['num_channels'], config['kernel_sizes']):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(config['num_channels'][-1], config['num_classes'])
    def forward(self, x):
        h = self.input_proj(x).transpose(1, 2)
        conv_out = self.conv_layers(h)
        return self.output_head(torch.max(conv_out, dim=2)[0])


# ===========================================
# 2. Data Loading & Preprocessing
# ===========================================

class SpeechCommandsSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        waveform, sample_rate, label, _, _ = self.subset[index]
        if self.transform:
            waveform, label = self.transform((waveform, label))
        return waveform, label
    def __len__(self):
        return len(self.subset)

def get_speech_commands_loaders(config):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']
    )

    def pad_and_transform(data):
        waveform, label = data
        target_len = config['sample_rate']
        padded_waveform = torch.zeros((1, target_len))
        if waveform.shape[1] > target_len:
            padded_waveform = waveform[:, :target_len]
        else:
            padded_waveform[:, :waveform.shape[1]] = waveform
        
        spectrogram = mel_spectrogram(padded_waveform)
        return spectrogram, label_to_idx[label]

    path = "./data"
    # *** FIX IS HERE ***
    os.makedirs(path, exist_ok=True) # Create directory if it doesn't exist
    dataset = torchaudio.datasets.SPEECHCOMMANDS(path, download=True)
    
    labels = sorted(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    all_labels = [item[2] for item in dataset]
    indices = [i for i, label in enumerate(all_labels) if label in label_to_idx]
    
    subset = Subset(dataset, indices)
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(subset, [0.8, 0.1, 0.1], generator=generator)

    train_dataset = SpeechCommandsSubset(train_set, transform=pad_and_transform)
    val_dataset = SpeechCommandsSubset(val_set, transform=pad_and_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader, len(labels)


# ===========================================
# 3. Training and Evaluation Loop
# ===========================================

def train_and_evaluate(model, train_loader, val_loader, config):
    model.to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']} [Training]")
        for specs, labels in train_loop:
            specs = specs.squeeze(1).transpose(1, 2).to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #train_loop.set_postfix(loss=loss.item())

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.squeeze(1).transpose(1, 2).to(config['device'])
                labels = labels.to(config['device'])
                outputs = model(specs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch} | Val Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}\n")
        
        if f1 > best_f1:
            best_f1 = f1
            final_metrics = {'Accuracy': accuracy, 'F1 Score': f1}
            
    return final_metrics

# ===========================================
# 4. Main Execution
# ===========================================

if __name__ == '__main__':
    set_seed(42)
    
    print("--- 1. Loading and Preprocessing Speech Commands Dataset ---")
    train_loader, val_loader, num_classes = get_speech_commands_loaders(CONFIG)
    print(f"Dataset ready. Number of classes: {num_classes}\n")
    
    results = {}
    models_to_test = {
        "Zarvan": AudioZarvanModel,
        "Transformer": AudioSimpleTransformerModel,
        "LSTM": AudioLSTMModel,
        "GRU": AudioGRUModel,
        "CNN_1D": AudioCNN1DModel,
    }
    
    for name, model_class in models_to_test.items():
        print(f"--- 2. Benchmarking {name} ---")
        
        model_config = MODEL_CONFIGS[name].copy()
        model_config['input_dim'] = CONFIG['n_mels']
        model_config['num_classes'] = num_classes
        
        model = model_class(model_config)
        
        start_time = time.time()
        metrics = train_and_evaluate(model, train_loader, val_loader, CONFIG)
        end_time = time.time()
        
        metrics['Training Time (s)'] = end_time - start_time
        results[name] = metrics

    print("\n" + "="*50)
    print("      Speech Commands Benchmark Final Results")
    print("="*50)
    
    results_df = pd.DataFrame(results).T
    print(results_df.to_string(float_format="%.4f"))
    print("="*50)
