import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import LongformerModel, LongformerConfig
import time
import math
from scipy.stats import entropy
import pandas as pd
import traceback
from collections import Counter
from datasets import load_dataset
import random
from sklearn.decomposition import PCA

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

def setup_text_data(num_difficulty_levels=10):
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", trust_remote_code=True)
    train_texts = [sample['text'] for sample in dataset['train']]
    train_labels = [sample['label'] for sample in dataset['train']]
    print("Building vocabulary...")
    vocab = TextVocabulary(train_texts)
    print("Creating difficulty buckets...")
    train_texts_by_len = sorted([(text, label) for text, label in zip(train_texts, train_labels)], key=lambda x: len(x[0].split()))
    difficulty_buckets = []
    total_samples = len(train_texts_by_len)
    samples_per_bucket = total_samples // num_difficulty_levels
    for i in range(num_difficulty_levels):
        start_idx = i * samples_per_bucket
        end_idx = (i + 1) * samples_per_bucket if i < num_difficulty_levels - 1 else total_samples
        bucket_data = train_texts_by_len[start_idx:end_idx]
        if not bucket_data: continue
        max_len = max(len(text.split()) for text, _ in bucket_data)
        if max_len == 0: continue
        difficulty_buckets.append(([d[0] for d in bucket_data], [d[1] for d in bucket_data], max_len))
    return vocab, difficulty_buckets

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
        return y, x, x

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
        return y, h_final, h_initial

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
        return y, h_final, h_initial

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
        return y, h_final, h_initial

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
        return y, h_final, h_initial

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
        return y, h_final, h_initial

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
        return y, h_final, h_initial

# ===========================================
# 3. IQ Calculation Functions
# ===========================================
def cka(X, Y):
    X, Y = X.float().reshape(X.shape[0], -1), Y.float().reshape(Y.shape[0], -1)
    if X.shape[0] < 2: return 0.0
    n = X.shape[0]
    H = torch.eye(n, device=X.device) - 1.0 / n
    K, L = torch.matmul(X, X.t()), torch.matmul(Y, Y.t())
    Kc, Lc = H @ K @ H, H @ L @ H
    cka_score = torch.sum(Kc * Lc) / (torch.norm(Kc, p='fro') * torch.norm(Lc, p='fro') + 1e-9)
    return max(0.0, cka_score.item())

def compute_processing_iq(model, inputs_task):
    if inputs_task.shape[0] < 2: return 0.0
    with torch.no_grad():
        inputs_task = inputs_task.to(next(model.parameters()).device)
        y, h_final, h_initial = model(inputs_task)
    
    itc = cka(h_initial, h_final)
    cci = cka(h_final, y)
    
    h_initial_np = h_initial.cpu().numpy().reshape(h_initial.shape[0], -1)
    h_final_np = h_final.cpu().numpy().reshape(h_final.shape[0], -1)
    
    n_components = min(h_initial_np.shape[0], h_initial_np.shape[1], h_final_np.shape[1])
    if n_components < 1: return 0.0
    
    pca_initial = PCA(n_components=n_components)
    pca_final = PCA(n_components=n_components)
    
    try:
        pca_initial.fit(h_initial_np)
        pca_final.fit(h_final_np)
        explained_variance_initial = np.sum(pca_initial.explained_variance_ratio_)
        explained_variance_final = np.sum(pca_final.explained_variance_ratio_)
        rp = explained_variance_final / (explained_variance_final + explained_variance_initial + 1e-9)
    except:
        rp = 0.0
    
    return itc * cci * rp

def compute_stability(iq_scores):
    if not iq_scores or len(iq_scores) < 2: return 0.0
    iq_scores = np.array(iq_scores)
    mean_iq = np.mean(iq_scores)
    std_iq = np.std(iq_scores)
    if mean_iq < 1e-6: return 0.0
    stability = 1.0 - (std_iq / (mean_iq + 1e-6))
    return max(0.0, stability)

def compute_lsi(losses):
    if not losses: return 0.0
    losses = np.array(losses)
    L_0 = losses[0]
    T = len(losses)
    if L_0 < 1e-6: return 0.0
    error_reduction = [(L_0 - losses[t]) / (t + 1) for t in range(T)]
    return np.sum(error_reduction)

def compute_eci(model, train_loader, device):
    model.eval()
    total_acc, count = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            if x_batch.shape[0] < 2: continue
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _, _ = model(x_batch)
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == y_batch).float().mean().item()
            count += 1
    accuracy = total_acc / count if count > 0 else 0.0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params == 0: return 0.0
    return accuracy / (num_params / 1e6)

def compute_esi(model, batch_size, vocab_size, device, model_name):
    model.to(device).eval()
    times = []
    seq_lengths = [64, 512, 2048]
    batch_size = min(batch_size, 8)
    
    for seq_len in seq_lengths:
        try:
            test_data = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            with torch.no_grad():
                for _ in range(50): model(test_data)  # Warm-up iterations
            iterations = 0
            start_time = time.time()
            duration = 10.0  # Measurement duration
            while time.time() - start_time < duration:
                with torch.no_grad(): model(test_data)
                iterations += 1
            end_time = time.time()
            time_per_batch = (end_time - start_time) / iterations if iterations > 0 else float('inf')
            time_per_seq = time_per_batch / batch_size
            times.append(time_per_seq)
        except RuntimeError:
            times.append(max(times) * 2 if times else 1.0)
    
    if len(times) < 3: return 0.0
    
    m_12 = abs(times[1] - times[0]) / math.log(seq_lengths[1] / seq_lengths[0])  # Use absolute value
    m_32 = abs(times[2] - times[1]) / math.log(seq_lengths[2] / seq_lengths[1])  # Use absolute value
    
    slope_ratio = m_32 / m_12 if m_12 != 0 else 1.0
    t_2 = max(times[1], 1e-5)
    
    esi = 1.0 / (slope_ratio * t_2 + 1e-6)
    print(f"Times t_1: {times[0]:.6f}, t_2: {times[1]:.6f}, t_3: {times[2]:.6f}, Slope m_1/2: {m_12:.6f}, Slope m_3/2: {m_32:.6f}, Ratio (m_3/2 / m_1/2): {slope_ratio:.4f}, ESI: {esi:.4f} for {model_name}")
    return max(0.0, esi)

def compute_ari_improved(model, inputs, labels, vocab, noise_levels=[0.05, 0.1, 0.2]):
    device = next(model.parameters()).device
    model.to(device).eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs, _, _ = model(inputs)
        base_loss = F.cross_entropy(outputs, labels).item()

    ari_scores = []
    for noise_level in noise_levels:
        perturbed_inputs = inputs.clone()
        for i in range(inputs.size(0)):
            seq_len = (inputs[i] != 0).sum().item()
            if seq_len == 0:
                continue
            num_to_replace = int(seq_len * noise_level)
            if num_to_replace == 0:
                continue
            indices_to_replace = torch.randperm(seq_len)[:num_to_replace]
            for idx in indices_to_replace:
                perturbed_inputs[i, idx] = torch.randint(2, len(vocab.vocab), (1,), device=device)
        with torch.no_grad():
            adv_outputs, _, _ = model(perturbed_inputs)
            adv_loss = F.cross_entropy(adv_outputs, labels).item()
        score = base_loss / (adv_loss + 1e-6)
        ari_scores.append(score)
    
    return np.sum(ari_scores) if ari_scores else 0.0

def compute_gti(model, train_loader, test_loader, device):
    model.to(device).eval()
    train_loss, test_loss, train_batches, test_batches = 0, 0, 0, 0
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            if x_batch.shape[0] < 2: continue
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _, _ = model(x_batch)
            train_loss += F.cross_entropy(outputs, y_batch).item()
            train_batches += 1
        for x_batch, y_batch in test_loader:
            if x_batch.shape[0] < 2: continue
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _, _ = model(x_batch)
            test_loss += F.cross_entropy(outputs, y_batch).item()
            test_batches += 1
    train_loss /= train_batches if train_batches > 0 else 1
    test_loss /= test_batches if test_batches > 0 else 1
    return train_loss / (test_loss + 1e-6)

def compute_onr(model, inputs):
    device = next(model.parameters()).device
    model.to(device).eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs, _, _ = model(inputs)
    num_bins = 50
    outputs_np = outputs.cpu().numpy().flatten()
    hist_y, _ = np.histogram(outputs_np, bins=num_bins, range=(-10, 10), density=True)
    prob_dist = hist_y / (np.sum(hist_y) + 1e-9)
    entropy_score = entropy(prob_dist + 1e-9)
    mean_magnitude = torch.mean(torch.abs(outputs)).item()
    return entropy_score * mean_magnitude

def compute_reasoning_index(model, inputs, labels, config):
    device = next(model.parameters()).device
    model.to(device).eval()
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        outputs_single, _, h_initial = model(inputs)
        loss_single = F.cross_entropy(outputs_single, labels).item()
        if not hasattr(model, 'fc_out') and not hasattr(model, 'output_head'): return 0.0
        
        projection_matrix = model.fc_out.weight if hasattr(model, 'fc_out') else model.output_head.weight
        projected_conclusion = F.softmax(outputs_single, dim=1) @ projection_matrix
        
        if projected_conclusion.shape[-1] != config['embed_dim']:
            projection_layer = nn.Linear(projected_conclusion.shape[-1], config['embed_dim'], device=device)
            projected_thought = projection_layer(projected_conclusion)
        else:
            projected_thought = projected_conclusion

        refined_embeddings = h_initial + projected_thought.unsqueeze(1)
        outputs_multi, _, _ = model(x=None, inputs_embeds=refined_embeddings)
        loss_multi = F.cross_entropy(outputs_multi, labels).item()

    return max(0.0, (loss_single - loss_multi) / (loss_single + loss_multi + 1e-6))

# ===========================================
# 4. Model Creators & Evaluation Function
# ===========================================
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

def evaluate_model(name, config, vocab, difficulty_buckets, device):
    try:
        model = create_text_model(name, config).to(device)
        local_report = {}
        print(f"🚀 [Evaluating {name}] Started evaluation on cuda.")
        print(f"  - Starting main training for {name}...")
        train_texts, train_labels, max_len = difficulty_buckets[4]
        train_loader = DataLoader(IMDBTextDataset(train_texts, train_labels, vocab, max_len), batch_size=config['batch_size'], shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        training_losses = []
        model.train()
        for epoch in range(config['max_epochs']):
            for x_batch, y_batch in train_loader:
                if x_batch.shape[0] < 2: continue
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions, _, _ = model(x_batch)
                loss = F.cross_entropy(predictions, y_batch)
                if torch.isnan(loss) or torch.isinf(loss): continue
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                total_loss, count = 0, 0
                for x_batch, y_batch in train_loader:
                    if x_batch.shape[0] < 2: continue
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    predictions, _, _ = model(x_batch)
                    total_loss += F.cross_entropy(predictions, y_batch).item()
                    count += 1
                training_losses.append(total_loss / count if count > 0 else 0)
        
        local_report['LSI'] = compute_lsi(training_losses)
        model.eval()
        local_report['ECI'] = compute_eci(model, train_loader, device)
        local_report['ESI'] = compute_esi(model, config['batch_size'], config['vocab_size'], device, name)
        
        print(f"  - Part 1 Finished. LSI: {local_report['LSI']:.3f}, ECI: {local_report['ECI']:.3f}, ESI: {local_report['ESI']:.3f}")

        iq_curve, ari_scores, rei_scores, onr_scores = [], [], [], []
        for texts, labels, max_len in difficulty_buckets:
            loader = DataLoader(IMDBTextDataset(texts, labels, vocab, max_len), batch_size=config['batch_size'], shuffle=False)
            if len(loader) == 0: continue
            inputs, labels_tensor = next(iter(loader))
            inputs, labels_tensor = inputs.to(device), labels_tensor.to(device)
            iq_curve.append(compute_processing_iq(model, inputs))
            ari_scores.append(compute_ari_improved(model, inputs, labels_tensor, vocab))
            rei_scores.append(compute_reasoning_index(model, inputs, labels_tensor, config))
            onr_scores.append(compute_onr(model, inputs))

        local_report['Processing_IQ_Mean'] = np.mean(iq_curve) if iq_curve else 0.0
        local_report['ARI'] = np.mean(ari_scores) if ari_scores else 0.0
        local_report['REI'] = np.mean(rei_scores) if rei_scores else 0.0
        local_report['ONR'] = np.mean(onr_scores) if onr_scores else 0.0
        local_report['Stability'] = compute_stability(iq_curve)

        gti_train, gti_test = difficulty_buckets[2], difficulty_buckets[8]
        gti_train_loader = DataLoader(IMDBTextDataset(gti_train[0], gti_train[1], vocab, gti_train[2]), batch_size=config['batch_size'])
        gti_test_loader = DataLoader(IMDBTextDataset(gti_test[0], gti_test[1], vocab, gti_test[2]), batch_size=config['batch_size'])
        local_report['GTI'] = compute_gti(model, gti_train_loader, gti_test_loader, device)

        print(f"  - Part 2 Finished. IQ_Mean: {local_report['Processing_IQ_Mean']:.3f}, ARI: {local_report['ARI']:.3f}, ONR: {local_report['ONR']:.3f}")
        return local_report
    except Exception as e:
        print(f"❌ CRITICAL ERROR evaluating model {name}: {e}")
        traceback.print_exc()
        return None

# ===========================================
# 5. Plotting & Main Execution
# ===========================================
def plot_radar_chart(model_name, scores, labels, weights, filename):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    
    # Calculate weighted scores
    weighted_scores = [scores[i] * weights.get(labels[i], 0) for i in range(num_vars)]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, weighted_scores + [weighted_scores[0]], alpha=0.25, label=model_name)
    ax.plot(angles, weighted_scores + [weighted_scores[0]], linewidth=2, label=model_name)
    
    # Dynamically set scale based on data
    max_score = max(weighted_scores) if weighted_scores else 0
    y_limit = max_score * 1.15  # Add 15% padding to the top
    if y_limit == 0: y_limit = 1 # Avoid zero limit

    ax.set_ylim(0, y_limit)
    num_ticks = 6 # Number of rings
    y_ticks = np.linspace(0, y_limit, num_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks], fontsize=10)
    
    # Set x-axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=12)
    
    plt.title(f"IQ Profile for {model_name}", size=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_radar_chart(final_results, weights, filename):
    num_vars = len(weights)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    labels = list(weights.keys())
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Color palette and line styles
    colors = plt.cm.tab10(np.linspace(0, 1, len(final_results)))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.'][:len(final_results)]
    
    # First, find the overall maximum score for dynamic scaling
    max_overall_score = 0
    all_model_weighted_scores = []
    for model_name, metrics in final_results.items():
        raw_scores = [metrics.get(k, 0.0) for k in labels]
        weighted_scores = [raw_scores[j] * weights.get(labels[j], 0) for j in range(num_vars)]
        all_model_weighted_scores.append(weighted_scores)
        current_max = max(weighted_scores) if weighted_scores else 0
        if current_max > max_overall_score:
            max_overall_score = current_max

    # Now, plot each model using the pre-calculated weighted scores
    for i, (model_name, weighted_scores) in enumerate(zip(final_results.keys(), all_model_weighted_scores)):
        ax.fill(angles, weighted_scores + [weighted_scores[0]], alpha=0.2, color=colors[i], label=model_name)
        ax.plot(angles, weighted_scores + [weighted_scores[0]], linewidth=2, linestyle=linestyles[i], color=colors[i])
    
    # Dynamically set scale based on all models' data
    y_limit = max_overall_score * 1.15  # Add 15% padding
    if y_limit == 0: y_limit = 1 # Avoid zero limit

    ax.set_ylim(0, y_limit)
    num_ticks = 6 # Number of rings
    y_ticks = np.linspace(0, y_limit, num_ticks)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks], fontsize=10)
    
    # Set x-axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=12)
    
    plt.title("IQ Profiles for All Models", size=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3), fontsize=10)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 Starting Text-IQ Benchmark (Single Run)...")
    vocab, difficulty_buckets = setup_text_data()

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
    report_card = {}

    for name in model_names:
        result = evaluate_model(name, config, vocab, difficulty_buckets, device)
        if result:
            report_card[name] = result
            print(f"🎉 Successfully completed evaluation for {name}.\n")

    print("\n📊 Final Ranking and Visualization...")
    if not report_card: 
        print("No models were successfully evaluated.")
        return

    weights = {
        # High Importance -> Target Impact: 99
        'IQ': 359.0,
        'ARI': 34.0,
        'LSI': 187.0,

        # Medium Importance -> Target Impact: 66
        'ECI': 174.0,
        'GTI': 50.0,
        'ONR': 11.0,

        # Low Importance -> Target Impact: 33
        'Stability': 35.0,
        'ESI': 0.043,
        'REI': 126.0
    }

    final_results = {}
    labels = ['IQ', 'Stability', 'ARI', 'LSI', 'ESI', 'ECI', 'GTI', 'ONR', 'REI']

    for name, metrics in report_card.items():
        scores_dict = {
            'IQ': metrics.get('Processing_IQ_Mean', 0.0), 
            'Stability': metrics.get('Stability', 0.0),
            'ARI': metrics.get('ARI', 0.0), 
            'LSI': metrics.get('LSI', 0.0), 
            'ESI': metrics.get('ESI', 0.0), 
            'ECI': metrics.get('ECI', 0.0), 
            'GTI': metrics.get('GTI', 0.0), 
            'ONR': metrics.get('ONR', 0.0),
            'REI': metrics.get('REI', 0.0)
        }
        final_score = (sum(weights[k] * v for k, v in scores_dict.items()) / sum(weights.values())) * 160
        final_results[name] = {'Final Score': final_score, **scores_dict}
        
        scores = [scores_dict[k] for k in labels]
        plot_radar_chart(name, scores, labels, weights, f"text_iq_profile_{name}.png")

    # Plot combined radar chart
    plot_combined_radar_chart(final_results, weights, "text_iq_profile_combined.png")

    if final_results:
        df = pd.DataFrame.from_dict(final_results, orient='index').sort_values(by="Final Score", ascending=False)
        df = df.rename(columns={'IQ': 'Proc. IQ', 'ESI': 'ESI', 'ONR': 'ONR', 'REI': 'REI'})
        print("\n" + "="*25 + " FINAL TEXT-IQ REPORT CARD " + "="*25)
        print(df.to_markdown(floatfmt=".4f"))
        print("="*70 + "\n")
        df.to_csv('text_iq_report.csv')
        print("  - Report card and radar charts saved successfully.")

if __name__ == '__main__':

    main()
