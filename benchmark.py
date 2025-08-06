# ============================================================================
#
#       IQ-NET: Holistic Aptitude Profiler (Version 16.1 - Full & Readable)
#
# This is the complete, unabridged, and stable version of the code for benchmarking
# 6 neural network models and generating radar charts for their performance across
# 15 metrics. It is designed for clarity, reproducibility, and GitHub presentation.
# The radar charts visualize each model’s profile using abbreviations: RSN (Reasoning),
# MEM (Memory), SCL (Scalability), ROB (Robustness), GEN (Generalization),
# HEAD (Learning Headroom), PAR (Parameter), INTP (Interpretability),
# UNC (Uncertainty), CONT (Continual Learning), SPF (Spatial Focus),
# PAT (Pattern Invariance), FREQ (Frequency Detection), RHY (Rhythm Comprehension),
# TRAJ (Trajectory Prediction).
#
# Dependencies: torch, numpy, scipy, pandas, matplotlib, cv2 (optional for image/video probes)
# Usage: Run the script to profile models and generate radar charts saved as PNG files.
#
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import math
from scipy.stats import entropy
from scipy.ndimage import rotate
import pandas as pd
import random
import copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import pi

# ===========================================
# 0. Reproducibility & Configuration
# ===========================================

def set_seed(seed_value=42):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

PROFILER_CONFIG = {
    'vocab_size': 100,
    'sequence_length': 128,
    'num_samples': 1200,
    'batch_size': 32,
    'probe_training_steps': 700,
    'learning_rate': 1e-3,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'embed_dim': 64,
    'hidden_dim': 128,
    'num_layers': 2,
    'num_heads': 4,
    'num_classes': 10,
    'dropout': 0.1,
    'image_size': 32,
    'audio_signal_length': 2048,
    'video_num_frames': 16,
}

# ===========================================
# 1. Model Implementations
# ===========================================

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
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

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embed_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_dim'],
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        self.output_head = nn.Linear(config['embed_dim'], config['num_classes'])
    
    def forward(self, x, embeds=None, return_features=False):
        h = self.embedding(x) if embeds is None else embeds
        h = self.pos_encoder(h)
        features = self.transformer_encoder(h)
        if return_features:
            return features
        return self.output_head(features)

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.lstm = nn.LSTM(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])
    
    def forward(self, x, embeds=None, return_features=False):
        h_init = self.embedding(x) if embeds is None else embeds
        features, _ = self.lstm(h_init)
        if return_features:
            return features
        return self.output_head(features)

# --- Zarvan Components ---
class _HolisticContextExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, _ = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=-1)
        head_outputs = torch.sum(weights * v, dim=2)
        return self.combine(head_outputs.reshape(B, self.embed_dim))

class _AssociativeContextExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj = nn.Linear(embed_dim, 1)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        return torch.sum(weights * v, dim=1)

class _ZarvanBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.holistic_ctx = _HolisticContextExtractor(embed_dim, num_heads)
        self.associative_ctx = _AssociativeContextExtractor(embed_dim)
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim * 2))
        self.update_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, x):
        B, S, E = x.shape
        q_holistic = self.holistic_ctx(x)
        q_associative = self.associative_ctx(x)
        q_holistic_exp = q_holistic.unsqueeze(1).expand(-1, S, -1)
        q_associative_exp = q_associative.unsqueeze(1).expand(-1, S, -1)
        gate_input = torch.cat([x, q_holistic_exp, q_associative_exp], dim=-1)
        input_gate, forget_gate = self.gate_net(gate_input).chunk(2, dim=-1)
        gated_x = torch.sigmoid(input_gate) * x + torch.sigmoid(forget_gate) * self.update_proj(x)
        return self.norm(x + self.ffn(gated_x))

class ZarvanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        self.layers = nn.ModuleList([_ZarvanBlock(config['embed_dim'], config['hidden_dim'], config['num_heads']) for _ in range(config['num_layers'])])
        self.output_head = nn.Linear(config['embed_dim'], config['num_classes'])

    def forward(self, x, embeds=None, return_features=False):
        h = self.pos_encoder(self.embedding(x) if embeds is None else embeds)
        for layer in self.layers:
            h = layer(h)
        if return_features:
            return h
        return self.output_head(h)

class GRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.gru = nn.GRU(
            input_size=config['embed_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, x, embeds=None, return_features=False):
        h_init = self.embedding(x) if embeds is None else embeds
        features, _ = self.gru(h_init)
        if return_features:
            return features
        return self.output_head(features)

# --- TCN Components ---
class _CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv = weight_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding != 0 else x

class _TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = _CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        num_channels = [config['embed_dim']] + [config['hidden_dim']] * config['num_layers']
        layers = []
        for i in range(config['num_layers']):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers.append(_TCNBlock(in_channels, out_channels, kernel_size=3, dilation=dilation_size, dropout=config['dropout']))
        self.tcn_net = nn.Sequential(*layers)
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, x, embeds=None, return_features=False):
        h = self.embedding(x) if embeds is None else embeds
        features = self.tcn_net(h.permute(0, 2, 1)).permute(0, 2, 1)
        if return_features:
            return features
        return self.output_head(features)

class CNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.conv1 = nn.Conv1d(in_channels=config['embed_dim'], out_channels=config['hidden_dim'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=config['hidden_dim'], out_channels=config['hidden_dim'], kernel_size=3, padding=1)
        self.dropout = nn.Dropout(config['dropout'])
        self.output_head = nn.Linear(config['hidden_dim'], config['num_classes'])

    def forward(self, x, embeds=None, return_features=False):
        h = self.embedding(x) if embeds is None else embeds
        h = h.permute(0, 2, 1)
        features = F.relu(self.conv1(h))
        features = self.dropout(features)
        features = F.relu(self.conv2(features))
        features = features.permute(0, 2, 1)
        if return_features:
            return features
        return self.output_head(features)

# ===========================================
# Data Generation
# ===========================================
def generate_composite_probe_data(num_samples, seq_len, vocab_size, num_classes,
                                  mem_difficulty_levels=[1, 2, 3, 5, 8],
                                  reas_difficulty_levels=[1, 2, 3]):
    PAD = 0
    COPY_TOKENS = list(range(10, 10 + num_classes))
    RECALL_TOKEN = 23
    FLIP_TOKEN, CYCLE_TOKEN, FLIP_A_TOKEN, FLIP_B_TOKEN = 20, 24, 25, 26
    DATA_TOKEN = 21

    sequences, mem_labels, reas_labels, mem_difficulty, reas_difficulty = [], [], [], [], []
    num_mem_levels = len(mem_difficulty_levels) if mem_difficulty_levels else 1
    num_reas_levels = len(reas_difficulty_levels) if reas_difficulty_levels else 1
    samples_per_combo = num_samples // (num_mem_levels * num_reas_levels)
    mem_levels_loop = mem_difficulty_levels if mem_difficulty_levels else [0]
    reas_levels_loop = reas_difficulty_levels if reas_difficulty_levels else [0]

    for mem_rep in mem_levels_loop:
        for reas_level in reas_levels_loop:
            for _ in range(samples_per_combo):
                seq = np.random.randint(30, vocab_size, size=seq_len)
                mem_label = np.full(seq_len, -100)
                reas_label = np.full(seq_len, -100)
                
                if mem_rep > 0:
                    target_token_vocab = random.choice(COPY_TOKENS)
                    indices = random.sample(range(5, seq_len // 2 - 5), mem_rep)
                    for idx in indices:
                        seq[idx] = target_token_vocab
                    recall_idx = seq_len - 5
                    seq[recall_idx] = RECALL_TOKEN
                    mem_label[recall_idx] = target_token_vocab - 10
                
                if reas_level > 0:
                    reas_start, reas_len = seq_len // 2, seq_len // 4
                    seq[reas_start : reas_start + reas_len] = PAD
                    if reas_level == 1:
                        state = 0
                        for i in range(reas_start, reas_start + reas_len):
                            if random.random() < 0.2:
                                seq[i], state = FLIP_TOKEN, 1 - state
                            else:
                                seq[i], reas_label[i] = DATA_TOKEN, state
                    elif reas_level == 2:
                        state = 0
                        for i in range(reas_start, reas_start + reas_len):
                            if random.random() < 0.2:
                                seq[i], state = CYCLE_TOKEN, (state + 1) % 3
                            else:
                                seq[i], reas_label[i] = DATA_TOKEN, state
                    elif reas_level == 3:
                        state_a, state_b = 0, 0
                        for i in range(reas_start, reas_start + reas_len):
                            r = random.random()
                            if r < 0.15:
                                seq[i], state_a = FLIP_A_TOKEN, 1 - state_a
                            elif r < 0.3:
                                seq[i], state_b = FLIP_B_TOKEN, 1 - state_b
                            else:
                                seq[i], reas_label[i] = DATA_TOKEN, state_a * 2 + state_b
                
                sequences.append(seq)
                mem_labels.append(mem_label)
                reas_labels.append(reas_label)
                mem_difficulty.append(mem_rep)
                reas_difficulty.append(reas_level)

    return {
        'sequences': torch.LongTensor(np.array(sequences)),
        'memory_labels': torch.LongTensor(np.array(mem_labels)),
        'reasoning_labels': torch.LongTensor(np.array(reas_labels)),
        'memory_difficulty': torch.LongTensor(np.array(mem_difficulty)),
        'reasoning_difficulty': torch.LongTensor(np.array(reas_difficulty))
    }

# ===========================================
# Profiler Implementation
# ===========================================
class HolisticAptitudeProfiler:
    def __init__(self, model_class, model_name, base_config):
        self.model_class = model_class
        self.model_name = model_name
        self.config = base_config
        self.device = base_config['device']
        set_seed(42)
        print(f"--- Profiling {self.model_name} ---")
        
        img_s = self.config['image_size']
        final_img_feature_map_size = img_s // 4
        img_cnn_out_features = 16 * final_img_feature_map_size * final_img_feature_map_size
        self.image_perception_head = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(img_cnn_out_features, self.config['embed_dim'])
        ).to(self.device)

        audio_frame_size = 256
        final_audio_feature_map_size = audio_frame_size // 4
        audio_cnn_out_features = 16 * final_audio_feature_map_size
        self.audio_perception_head = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(audio_cnn_out_features, self.config['embed_dim'])
        ).to(self.device)
        
        if self.model_name in ['Transformer', 'Zarvan']:
            feature_dim = self.config['embed_dim']
        else:
            feature_dim = self.config['hidden_dim']
        
        self.regression_head = nn.Linear(feature_dim, 2).to(self.device)
        self.multidomain_classification_head = nn.Linear(feature_dim, self.config['num_classes']).to(self.device)

    def get_profile(self):
        profile = {}
        probe_data = generate_composite_probe_data(
            self.config['num_samples'], self.config['sequence_length'],
            self.config['vocab_size'], self.config['num_classes']
        )
        
        print("Running static tests...")
        profile['Parameter_Score'] = self._compute_parameter_score()
        print(f"  > Parameter_Score calculated: {profile['Parameter_Score']:.4f}")
        profile['Scalability_Score'] = self._compute_scalability_score()
        print(f"  > Scalability_Score calculated: {profile['Scalability_Score']:.4f}")
        
        print("Running probe training and evaluation...")
        model = self.model_class(self.config).to(self.device)
        probe_results = self._run_probe_training(model, probe_data)
        model.eval()
        
        profile['Memory_Score'] = self._compute_memory_score(model, probe_data)
        profile['Reasoning_Score'] = self._compute_reasoning_score(model, probe_data)
        
        profile['Generalization_Score'] = self._compute_generalization_score(probe_results)
        print(f"  > Generalization_Score calculated: {profile['Generalization_Score']:.4f}")
        
        profile['Robustness_Score'] = self._compute_robustness_score(model, probe_data, probe_results)
        
        profile['Learning_Headroom'] = probe_results['learning_headroom']
        print(f"  > Learning_Headroom calculated: {profile['Learning_Headroom']:.4f}")
        
        profile['Interpretability_Score'] = self._compute_interpretability_score(model, probe_data)
        print(f"  > Interpretability_Score calculated: {profile['Interpretability_Score']:.4f}")
        
        print("Running advanced probes...")
        profile['Uncertainty_Score'] = self._compute_uncertainty_score()
        print(f"  > Uncertainty_Score calculated: {profile['Uncertainty_Score']:.4f}")
        
        profile['Continual_Learning_Score'] = self._compute_continual_learning_score()
        print(f"  > Continual_Learning_Score calculated: {profile['Continual_Learning_Score']:.4f}")
        
        print("Running Image IQ Probes...")
        profile.update(self._compute_image_profile(model))
        
        print("Running Audio IQ Probes...")
        profile.update(self._compute_audio_profile(model))
        
        print("Running Video IQ Probes...")
        profile.update(self._compute_video_profile(model))
        
        print(f"Profile for {self.model_name} complete.\n")
        return profile

    def _compute_image_profile(self, model):
        image_scores = {}
        image_scores['Spatial_Focus_Score'] = self._compute_spatial_focus_score(model)
        image_scores['Pattern_Invariance_Score'] = self._compute_pattern_invariance_score(model)
        return image_scores

    def _generate_spatial_focus_data(self, level, batch_size=32, img_size=32):
        images = np.random.rand(batch_size, img_size, img_size, 3).astype(np.float32) * 0.1
        coords = np.zeros((batch_size, 2), dtype=np.float32); shape_color = np.array([0.9, 0.1, 0.1])
        for i in range(batch_size):
            shape_size = img_size // 4 if level == 1 else img_size // 8
            if level == 2:
                for _ in range(5): x1,y1,x2,y2=np.random.randint(0,img_size,4); cv2.line(images[i], (x1,y1), (x2,y2), np.random.rand(3), 1)
            elif level == 3:
                for _ in range(5): dx, dy = np.random.randint(shape_size, img_size-shape_size,2); d_size=shape_size//2; images[i, dy-d_size:dy+d_size, dx-d_size:dx+d_size] = shape_color
            cx, cy = np.random.randint(shape_size, img_size - shape_size, 2); start_y, end_y = max(0, cy-shape_size), min(img_size, cy+shape_size); start_x, end_x = max(0, cx-shape_size), min(img_size, cx+shape_size)
            images[i, start_y:end_y, start_x:end_x] = shape_color; coords[i] = [cx / img_size, cy / img_size]
        return torch.from_numpy(images).permute(0, 3, 1, 2), torch.from_numpy(coords)

    def _compute_spatial_focus_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}; level_scores = {}; print("  > Calculating Spatial Focus Score...")
        for level, name in levels.items():
            images, true_coords = self._generate_spatial_focus_data(level, self.config['batch_size'], self.config['image_size']); images, true_coords = images.to(self.device), true_coords.to(self.device)
            with torch.no_grad():
                embeddings = self.image_perception_head(images).unsqueeze(1); model_features = model(x=None, embeds=embeddings, return_features=True)
                predicted_coords = torch.sigmoid(self.regression_head(model_features[:, 0, :]))
            distance = torch.sqrt(((predicted_coords - true_coords)**2).sum(dim=1)).mean().item(); score = math.exp(-10 * distance)
            level_scores[level] = score; print(f"    - Level {level} ({name}) Score: {score:.4f}")
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items()); final_score = weighted_sum / sum(weights.values()); print(f"  > Final Spatial_Focus_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _generate_pattern_invariance_data(self, level, batch_size=32, img_size=32):
        images, labels = np.zeros((batch_size,img_size,img_size,1), dtype=np.float32), np.zeros(batch_size, dtype=int); pattern = np.zeros((7,7)); pattern[1:6,1]=1; pattern[5,1:4]=1
        for i in range(batch_size):
            img = np.zeros((img_size, img_size)); is_target = random.random() > 0.5; current_pattern = pattern if is_target else np.rot90(pattern, k=2); p_size_h, p_size_w = 7, 7
            if level >= 2: new_size = random.randint(5,12); current_pattern=cv2.resize(current_pattern, (new_size,new_size)); p_size_h,p_size_w = current_pattern.shape
            if level >= 3: angle = random.uniform(-45,45); current_pattern=rotate(current_pattern,angle,reshape=True,mode='constant',cval=0.0); p_size_h,p_size_w=current_pattern.shape
            if p_size_h>=img_size or p_size_w>=img_size: images[i,:,:,0]=np.zeros((img_size,img_size)); labels[i]=0; continue
            x, y = np.random.randint(0, img_size-p_size_w), np.random.randint(0, img_size-p_size_h)
            img[y:y+p_size_h, x:x+p_size_w] = current_pattern; images[i,:,:,0]=img; labels[i]=1 if is_target else 0
        return torch.from_numpy(images).permute(0, 3, 1, 2).repeat(1,3,1,1), torch.from_numpy(labels)

    def _compute_pattern_invariance_score(self, model):
        levels, weights = {1:'translation', 2:'scale', 3:'rotation'}, {1:1, 2:2, 3:4}; level_scores={}; print("  > Calculating Pattern Invariance Score...")
        for level, name in levels.items():
            images, labels = self._generate_pattern_invariance_data(level, self.config['batch_size']*4, self.config['image_size'])
            dataset = TensorDataset(images, labels); loader = DataLoader(dataset, batch_size=self.config['batch_size']); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labs in loader:
                    imgs, labs = imgs.to(self.device), labs.to(self.device); embeddings=self.image_perception_head(imgs).unsqueeze(1)
                    model_features = model(x=None, embeds=embeddings, return_features=True); logits=self.multidomain_classification_head(model_features[:, 0, :])
                    preds = torch.argmax(logits, dim=1); correct+=(preds==labs).sum().item(); total+=labs.size(0)
            accuracy=correct/total if total>0 else 0; level_scores[level]=accuracy; print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l, 0)*s for l, s in level_scores.items()); final_score = weighted_sum / sum(weights.values()); print(f"  > Final Pattern_Invariance_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _compute_audio_profile(self, model):
        audio_scores = {}; audio_scores['Frequency_Detection_Score'] = self._compute_frequency_detection_score(model); audio_scores['Rhythm_Comprehension_Score'] = self._compute_rhythm_comprehension_score(model)
        return audio_scores

    def _generate_frequency_detection_data(self, level, batch_size=32, sr=8000, length=2048):
        signals, labels = np.zeros((batch_size,length), dtype=np.float32), np.zeros(batch_size, dtype=int); freqs=[261,329,392,440]; t=np.linspace(0,length/sr,length,endpoint=False)
        for i in range(batch_size):
            target_freq_idx=random.randint(0,len(freqs)-1); target_freq=freqs[target_freq_idx]; amp,noise_amp=(0.8,0.2) if level==1 else((0.2,0.2) if level==2 else(0.5,0.2))
            signal=np.random.randn(length)*noise_amp; signal+=np.sin(2*np.pi*target_freq*t)*amp
            if level==3: distractor_freq=random.choice([f for f in freqs if f!=target_freq]); signal+=np.sin(2*np.pi*distractor_freq*t)*(amp*0.8)
            signals[i,:]=signal; labels[i]=target_freq_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_frequency_detection_score(self, model):
        levels, weights = {1:'easy',2:'medium',3:'hard'}, {1:1,2:2,3:4}; level_scores={}; frame_size=256; print("  > Calculating Frequency Detection Score...")
        for level, name in levels.items():
            signals, labels = self._generate_frequency_detection_data(level, self.config['batch_size']*4)
            dataset = TensorDataset(signals, labels); loader = DataLoader(dataset, batch_size=self.config['batch_size']); correct, total = 0, 0
            with torch.no_grad():
                for sigs, labs in loader:
                    sigs, labs = sigs.to(self.device), labs.to(self.device); frames = sigs.unfold(1,frame_size,frame_size//2).contiguous(); B,N,F = frames.shape
                    framed_sigs=frames.view(B*N,F).unsqueeze(1); embeddings=self.audio_perception_head(framed_sigs).view(B,N,-1)
                    model_features = model(x=None, embeds=embeddings, return_features=True); pooled_output = self.multidomain_classification_head(model_features.mean(dim=1))
                    preds = torch.argmax(pooled_output, dim=1); correct+=(preds==labs).sum().item(); total+=labs.size(0)
            accuracy=correct/total if total>0 else 0; level_scores[level]=accuracy; print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l,0)*s for l,s in level_scores.items()); final_score = weighted_sum / sum(weights.values()); print(f"  > Final Frequency_Detection_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _generate_rhythm_comprehension_data(self, level, batch_size=32, length=2048):
        signals = np.zeros((batch_size,length), dtype=np.float32); labels = np.zeros(batch_size, dtype=int); patterns=[[1,0,1,0],[1,1,0,0],[1,0,0,1]]
        for i in range(batch_size):
            pattern_idx=random.randint(0,len(patterns)-1); pattern=patterns[pattern_idx]
            if level>=2: pattern=pattern+random.choice(patterns)
            pulse_len=100; signal=np.zeros(length); pos=50
            for beat in pattern:
                jitter = int(random.uniform(-20,20)) if level==3 else 0
                if beat==1 and pos+pulse_len+jitter<length and pos+jitter>=0: signal[pos+jitter:pos+jitter+pulse_len]=np.sin(np.linspace(0,3*np.pi,pulse_len))
                pos+=pulse_len*2
            signals[i,:]=signal+np.random.randn(length)*0.1; labels[i]=pattern_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_rhythm_comprehension_score(self, model):
        levels, weights = {1:'easy',2:'medium',3:'hard'}, {1:1,2:2,3:4}; level_scores={}; frame_size=256; print("  > Calculating Rhythm Comprehension Score...")
        for level, name in levels.items():
            signals, labels = self._generate_rhythm_comprehension_data(level, self.config['batch_size']*4)
            dataset = TensorDataset(signals, labels); loader = DataLoader(dataset, batch_size=self.config['batch_size']); correct, total = 0, 0
            with torch.no_grad():
                for sigs, labs in loader:
                    sigs, labs = sigs.to(self.device), labs.to(self.device); frames = sigs.unfold(1,frame_size,frame_size//2).contiguous(); B,N,F = frames.shape
                    framed_sigs=frames.view(B*N,F).unsqueeze(1); embeddings=self.audio_perception_head(framed_sigs).view(B,N,-1)
                    model_features = model(x=None, embeds=embeddings, return_features=True); pooled_output = self.multidomain_classification_head(model_features.mean(dim=1))
                    preds = torch.argmax(pooled_output, dim=1); correct+=(preds==labs).sum().item(); total+=labs.size(0)
            accuracy=correct/total if total>0 else 0; level_scores[level]=accuracy; print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l,0)*s for l,s in level_scores.items()); final_score = weighted_sum / sum(weights.values()); print(f"  > Final Rhythm_Comprehension_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _compute_video_profile(self, model):
        video_scores={}; video_scores['Trajectory_Prediction_Score']=self._compute_trajectory_prediction_score(model); return video_scores

    def _generate_trajectory_prediction_data(self, level, batch_size=32, num_frames=16, img_size=32):
        videos=np.zeros((batch_size,num_frames,img_size,img_size,3),dtype=np.float32); final_coords=np.zeros((batch_size,2),dtype=np.float32); shape_size=img_size//8
        for i in range(batch_size):
            start_x,start_y=np.random.randint(shape_size,img_size//2,2)
            if level==1: vx,vy,gx,gy = *np.random.uniform(0.5,2.0,2),0,0
            elif level==2: vx,vy,gx,gy=np.random.uniform(2.0,3.0),np.random.uniform(-1.0,-0.5),0,0.1
            else: vx,vy,gx,gy = *np.random.uniform(1.5,2.5,2),0,0
            x,y = float(start_x), float(start_y)
            for t in range(num_frames):
                x,y=x+vx,y+vy; vx,vy=vx+gx,vy+gy
                if level==3:
                    if not(shape_size<x<img_size-shape_size): vx*=-1
                    if not(shape_size<y<img_size-shape_size): vy*=-1
                if t<10:
                    ix,iy=int(x),int(y)
                    if 0<=ix<img_size and 0<=iy<img_size:
                        start_y_c,end_y_c=max(0,iy-shape_size),min(img_size,iy+shape_size); start_x_c,end_x_c=max(0,ix-shape_size),min(img_size,ix+shape_size)
                        videos[i,t,start_y_c:end_y_c,start_x_c:end_x_c] = [0.1,0.9,0.1]
            final_coords[i]=np.clip([x/img_size,y/img_size],0,1)
        return torch.from_numpy(videos).permute(0,1,4,2,3), torch.from_numpy(final_coords)

    def _compute_trajectory_prediction_score(self, model):
        levels,weights={1:'linear',2:'parabolic',3:'bounce'},{1:1,2:2,3:4}; level_scores={}; print("  > Calculating Trajectory Prediction Score...")
        for level,name in levels.items():
            videos,true_coords=self._generate_trajectory_prediction_data(level, self.config['batch_size'], self.config['video_num_frames'], self.config['image_size']); videos,true_coords=videos.to(self.device),true_coords.to(self.device); B,T,C,H,W=videos.shape
            with torch.no_grad():
                video_frames=videos.reshape(B*T,C,H,W); frame_embeddings=self.image_perception_head(video_frames).view(B,T,-1)
                model_features=model(x=None,embeds=frame_embeddings,return_features=True)
                predicted_coords=torch.sigmoid(self.regression_head(model_features[:,-1,:]))
            distance=torch.sqrt(((predicted_coords-true_coords)**2).sum(dim=1)).mean().item(); score=math.exp(-10*distance)
            level_scores[level]=score; print(f"    - Level {level} ({name}) Score: {score:.4f}")
        weighted_sum=sum(weights.get(l,0)*s for l,s in level_scores.items()); final_score=weighted_sum/sum(weights.values()); print(f"  > Final Trajectory_Prediction_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _run_probe_training(self, model, probe_data):
        dataset = TensorDataset(probe_data['sequences'], probe_data['memory_labels'], probe_data['reasoning_labels']); loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        all_params = list(model.parameters()) + list(self.image_perception_head.parameters()) + list(self.audio_perception_head.parameters()) + list(self.regression_head.parameters()) + list(self.multidomain_classification_head.parameters())
        optimizer = optim.AdamW(all_params, lr=self.config['learning_rate']); loss_fn = nn.CrossEntropyLoss(ignore_index=-100); model.train()
        step, training_complete, probe_results = 0, False, {'train_loss':[],'val_loss':0.0,'val_accuracy':0.0,'learning_headroom':0.0}
        while not training_complete:
            for seqs, mem_labels, reas_labels in loader:
                seqs, mem_labels, reas_labels = seqs.to(self.device), mem_labels.to(self.device), reas_labels.to(self.device); optimizer.zero_grad()
                logits = model(seqs)
                loss_mem = loss_fn(logits.view(-1, self.config['num_classes']), mem_labels.view(-1)); loss_reas = loss_fn(logits.view(-1, self.config['num_classes']), reas_labels.view(-1))
                total_loss = loss_mem + loss_reas
                if not torch.isnan(total_loss): total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                probe_results['train_loss'].append(total_loss.item()); step += 1
                if step >= self.config['probe_training_steps']: training_complete = True; break
        probe_results['learning_headroom'] = self._compute_learning_headroom(probe_results['train_loss']); model.eval()
        with torch.no_grad():
            correct_mem, total_mem, val_loss_sum, batches = 0, 0, 0, 0
            for seqs, mem_labels, reas_labels in loader:
                seqs, mem_labels, reas_labels = seqs.to(self.device), mem_labels.to(self.device), reas_labels.to(self.device); logits = model(seqs)
                preds, mask = torch.argmax(logits, dim=-1), mem_labels != -100
                correct_mem += (preds[mask] == mem_labels[mask]).sum().item(); total_mem += mask.sum().item()
                loss_mem, loss_reas = loss_fn(logits.view(-1, self.config['num_classes']), mem_labels.view(-1)), loss_fn(logits.view(-1, self.config['num_classes']), reas_labels.view(-1))
                if not torch.isnan(loss_mem) and not torch.isnan(loss_reas): val_loss_sum += (loss_mem + loss_reas).item(); batches += 1
            probe_results['val_accuracy'] = correct_mem/total_mem if total_mem > 0 else 0.0
            probe_results['val_loss'] = val_loss_sum/batches if batches > 0 else float('inf')
        return probe_results

    def _compute_parameter_score(self):
        model = self.model_class(self.config); return 1.0 / (1.0 + (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000.0))

    def _compute_scalability_score(self):
        times, seq_lengths = [], np.array([64, 128, 256]); model_instance = self.model_class(self.config).to(self.device).eval()
        with torch.no_grad():
            for sl in seq_lengths:
                try:
                    inp = torch.randint(0,self.config['vocab_size'], (self.config['batch_size']//2, int(sl)), device=self.device)
                    for _ in range(5): model_instance(inp)
                    start_time=time.time()
                    for _ in range(10): model_instance(inp)
                    times.append((time.time()-start_time)/10)
                except Exception: times.append(1e9)
        if any(t > 1e8 for t in times): return 0.0
        log_lengths, log_times = np.log(seq_lengths), np.log(np.array(times) + 1e-9)
        try: beta,_ = np.linalg.lstsq(np.vstack([log_lengths, np.ones(len(log_lengths))]).T, log_times, rcond=None)[0]
        except np.linalg.LinAlgError: beta = 5.0
        t_median = times[1]; s_complexity = math.exp(-0.7*max(0,beta-1.0)); s_speed = math.exp(-t_median/0.05)
        return s_complexity * s_speed

    def _compute_interpretability_score(self, model, probe_data):
        interpret_data=generate_composite_probe_data(self.config['batch_size'],self.config['sequence_length'],self.config['vocab_size'],self.config['num_classes'],mem_difficulty_levels=[1],reas_difficulty_levels=[]); loader=DataLoader(TensorDataset(*interpret_data.values()),batch_size=self.config['batch_size']); seqs,labels,_,_,_=next(iter(loader)); seqs,labels=seqs.to(self.device),labels.to(self.device)
        with torch.no_grad():
            logits=model(seqs); probs=F.softmax(logits,dim=-1); recall_mask=labels!=-100
            correct_class_indices=labels[recall_mask]; p_originals=probs[recall_mask].gather(1,correct_class_indices.unsqueeze(1)).squeeze()
        recall_token_id,copy_token_min_id,copy_token_max_id=23,10,10+self.config['num_classes']; signal_indices=[torch.where((seqs[i]>=copy_token_min_id)&(seqs[i]<copy_token_max_id))[0] for i in range(seqs.size(0))]; recall_indices=[torch.where(seqs[i]==recall_token_id)[0] for i in range(seqs.size(0))]; noise_impacts,signal_impacts=[],[]; num_noise_tokens_to_test=5
        for i in range(seqs.size(0)):
            if not hasattr(p_originals,"__len__") or i>=len(p_originals) or len(recall_indices[i])==0: continue
            p_original=p_originals[i].item()
            if p_original<0.5: continue
            signal_idxs_to_test=torch.cat([signal_indices[i],recall_indices[i]])
            if len(signal_idxs_to_test)==0: continue
            current_signal_impacts=[]
            for idx in signal_idxs_to_test:
                seq_perturbed=seqs[i].clone(); seq_perturbed[idx]=0
                with torch.no_grad(): p_new=F.softmax(model(seq_perturbed.unsqueeze(0)),dim=-1)[0,recall_indices[i][0],correct_class_indices[i]]; current_signal_impacts.append(p_original-p_new.item())
            if current_signal_impacts: signal_impacts.append(np.mean(current_signal_impacts))
            current_noise_impacts=[]
            all_indices=set(range(seqs.size(1))); signal_idx_set=set(signal_idxs_to_test.tolist()); population=list(all_indices-signal_idx_set)
            if len(population)<num_noise_tokens_to_test: continue
            noise_indices=random.sample(population,num_noise_tokens_to_test)
            for idx in noise_indices:
                seq_perturbed=seqs[i].clone(); seq_perturbed[idx]=0
                with torch.no_grad(): p_new=F.softmax(model(seq_perturbed.unsqueeze(0)),dim=-1)[0,recall_indices[i][0],correct_class_indices[i]]; current_noise_impacts.append(abs(p_original-p_new.item()))
            if current_noise_impacts: noise_impacts.append(np.mean(current_noise_impacts))
        if not signal_impacts or not noise_impacts: return 0.0
        mean_signal_impact=np.mean(signal_impacts); mean_noise_impact=np.mean(noise_impacts); focus_ratio=max(0,mean_signal_impact)/(mean_noise_impact+1e-9)
        return 1.0/(1.0+math.exp(-0.5*(math.log(focus_ratio+1)-math.log(10))))

    def _evaluate_multi_level(self, model, probe_data, task_type):
        if task_type=='memory': difficulty_tensor,labels_tensor,weights=probe_data['memory_difficulty'],probe_data['memory_labels'],{1:10,2:8,3:6,5:4,8:2}
        else: difficulty_tensor,labels_tensor,weights=probe_data['reasoning_difficulty'],probe_data['reasoning_labels'],{1:2,2:4,3:6}
        difficulty_levels=sorted(difficulty_tensor.unique().tolist()); correct_counts={level:0 for level in difficulty_levels}; total_counts={level:0 for level in difficulty_levels}
        dataset=TensorDataset(probe_data['sequences'],labels_tensor,difficulty_tensor); loader=DataLoader(dataset,batch_size=self.config['batch_size'])
        with torch.no_grad():
            for seqs, labels, difficulties in loader:
                seqs, labels=seqs.to(self.device),labels.to(self.device); preds=torch.argmax(model(seqs),dim=-1); mask=labels!=-100
                for level in difficulty_levels:
                    if level==0: continue
                    level_mask=(difficulties==level).unsqueeze(1)&mask.cpu(); correct_counts[level]+=(preds.cpu()[level_mask]==labels.cpu()[level_mask]).sum().item(); total_counts[level]+=level_mask.sum().item()
        return {level:correct_counts[level]/total_counts[level] if total_counts[level]>0 else 0 for level in difficulty_levels}, weights

    def _compute_memory_score(self, model, probe_data):
        accuracies,weights=self._evaluate_multi_level(model,probe_data,'memory'); print(f"  > Memory Accuracy per Repetition Level: { {k: round(v, 4) for k, v in accuracies.items()} }")
        weighted_sum=sum(weights.get(level,0)*accuracies.get(level,0) for level in weights); total_weight=sum(weights.values()); final_score=weighted_sum/total_weight if total_weight>0 else 0.0; print(f"  > Memory_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _compute_reasoning_score(self, model, probe_data):
        accuracies,weights=self._evaluate_multi_level(model,probe_data,'reasoning'); print(f"  > Reasoning Accuracy per Difficulty Level: { {k: round(v, 4) for k, v in accuracies.items()} }")
        normalized_accuracies={}; choices={1:2,2:3,3:4}
        for level,acc in accuracies.items():
            baseline=1.0/choices.get(level,2); normalized_accuracies[level]=max(0.0,(acc-baseline)/(1.0-baseline)) if (1.0-baseline)>0 else 0.0
        weighted_sum=sum(weights.get(level,0)*normalized_accuracies.get(level,0) for level in weights); total_weight=sum(weights.values()); final_score=weighted_sum/total_weight if total_weight>0 else 0.0; print(f"  > Reasoning_Score (Weighted) calculated: {final_score:.4f}"); return final_score

    def _compute_robustness_score(self, model, probe_data, pr):
        noise_levels,weights,level_scores=[0.05,0.15,0.30],{0.05:1,0.15:2,0.30:3}, {}; print(f"  > Robustness Score per Noise Level:")
        for level in noise_levels:
            noisy_sequences=probe_data['sequences'].clone(); mask=(noisy_sequences>=30); noise=torch.randint(30,self.config['vocab_size'],noisy_sequences.shape)
            perturb_mask=(torch.rand(noisy_sequences.shape)<level)&mask; noisy_sequences[perturb_mask]=noise[perturb_mask]
            noisy_dataset=TensorDataset(noisy_sequences,probe_data['memory_labels'],probe_data['reasoning_labels']); noisy_loader=DataLoader(noisy_dataset,batch_size=self.config['batch_size'])
            loss_fn,noisy_loss_sum,batches=nn.CrossEntropyLoss(ignore_index=-100),0,0
            with torch.no_grad():
                for seqs,mem_labels,reas_labels in noisy_loader:
                    seqs,mem_labels,reas_labels=seqs.to(self.device),mem_labels.to(self.device),reas_labels.to(self.device)
                    logits=model(seqs); loss_m,loss_r=loss_fn(logits.view(-1,self.config['num_classes']),mem_labels.view(-1)),loss_fn(logits.view(-1,self.config['num_classes']),reas_labels.view(-1))
                    if not torch.isnan(loss_m) and not torch.isnan(loss_r): noisy_loss_sum+=(loss_m+loss_r).item(); batches+=1
            noisy_loss=noisy_loss_sum/batches if batches>0 else float('inf'); scaler=max(0.0,(pr['val_accuracy']-0.5)*2)
            penalty_val=max(0,noisy_loss-pr['val_loss'])/(noisy_loss+pr['val_loss']+1e-9); penalty=math.sqrt(penalty_val) if penalty_val>0 else 0
            level_scores[level]=scaler*(1.0-penalty); print(f"    - Noise {int(level*100)}%: {level_scores[level]:.4f}")
        weighted_sum=sum(weights.get(level,0)*level_scores.get(level,0) for level in weights); total_weight=sum(weights.values()); final_score=weighted_sum/total_weight if total_weight>0 else 0.0; return final_score

    def _compute_generalization_score(self, pr):
        penalty_val=max(0,pr['val_loss']-np.mean(pr['train_loss']))/(pr['val_loss']+np.mean(pr['train_loss'])+1e-9); penalty=math.sqrt(penalty_val) if penalty_val>0 else 0; return pr['val_accuracy']*(1.0-penalty)

    def _compute_uncertainty_score(self):
        confusing_data=generate_composite_probe_data(self.config['batch_size'],self.config['sequence_length'],self.config['vocab_size'],self.config['num_classes'],reas_difficulty_levels=[])
        confusing_input=confusing_data['sequences'].to(self.device); confusing_input[confusing_input==23]=0; model=self.model_class(self.config).to(self.device).eval()
        with torch.no_grad():
            logits=model(confusing_input); probs=torch.softmax(logits,dim=-1); avg_entropy=-torch.sum(probs*torch.log(probs+1e-9),dim=-1).mean().item()
        max_entropy=np.log(self.config['num_classes']); return avg_entropy/max_entropy if max_entropy>0 else 0.0

    def _compute_continual_learning_score(self):
        temp_model=self.model_class(self.config).to(self.device); optimizer,loss_fn=optim.AdamW(temp_model.parameters(),lr=1e-3),nn.CrossEntropyLoss(ignore_index=-100)
        data_A=generate_composite_probe_data(256,self.config['sequence_length'],self.config['vocab_size'],self.config['num_classes'],mem_difficulty_levels=[3],reas_difficulty_levels=[])
        loader_A=DataLoader(TensorDataset(data_A['sequences'],data_A['memory_labels']),batch_size=32)
        data_B=generate_composite_probe_data(256,self.config['sequence_length'],self.config['vocab_size'],self.config['num_classes'],mem_difficulty_levels=[],reas_difficulty_levels=[1])
        loader_B=DataLoader(TensorDataset(data_B['sequences'],data_B['reasoning_labels']),batch_size=32)
        temp_model.train()
        for i,(seqs,labels) in enumerate(loader_A):
            if i>20: break
            seqs,labels=seqs.to(self.device),labels.to(self.device); optimizer.zero_grad(); loss=loss_fn(temp_model(seqs).view(-1,self.config['num_classes']),labels.view(-1)); loss.backward(); optimizer.step()
        with torch.no_grad(): acc_A1=self._evaluate_on_loader(temp_model,loader_A,'memory')
        temp_model.train()
        for i,(seqs,labels) in enumerate(loader_B):
            if i>20: break
            seqs,labels=seqs.to(self.device),labels.to(self.device); optimizer.zero_grad(); loss=loss_fn(temp_model(seqs).view(-1,self.config['num_classes']),labels.view(-1)); loss.backward(); optimizer.step()
        with torch.no_grad(): acc_A2=self._evaluate_on_loader(temp_model,loader_A,'memory')
        forgetting_factor=max(0,acc_A1-acc_A2)/(acc_A1+1e-9); return 1.0-forgetting_factor

    def _compute_learning_headroom(self,all_train_losses):
        if len(all_train_losses)<50: return 0.0
        last_losses=all_train_losses[-50:]; steps=np.arange(len(last_losses)); slope,_=np.polyfit(steps,last_losses,1)
        return 1.0/(1.0+math.exp(slope*100))

    def _evaluate_on_loader(self, model, loader_or_data, task_type):
        if isinstance(loader_or_data, dict):
            labels_tensor = loader_or_data['memory_labels'] if task_type=='memory' else loader_or_data['reasoning_labels']
            dataset = TensorDataset(loader_or_data['sequences'], labels_tensor); loader = DataLoader(dataset, batch_size=self.config['batch_size'])
        else: loader=loader_or_data
        correct,total=0,0
        with torch.no_grad():
            for batch in loader:
                seqs,labels=batch[0],batch[1]; seqs,labels=seqs.to(self.device),labels.to(self.device)
                preds=torch.argmax(model(seqs),dim=-1); mask=labels!=-100
                correct+=(preds[mask]==labels[mask]).sum().item(); total+=mask.sum().item()
        return correct/total if total>0 else 0.0

try:
    import cv2
except ImportError:
    print("Warning: OpenCV is not installed. Image/Video probes will fail.")
    print("Please install it using: pip install opencv-python")
    class DummyCV2:
        def line(self, *args, **kwargs): pass
        def resize(self, arr, size): return arr[:size[0], :size[1]]
    cv2 = DummyCV2()

# ===========================================
# 4. Main Execution and Visualization
# ===========================================

def plot_radar_charts(df):
    """
    Generates radar charts for model performance across 15 metrics.
    Creates individual charts for each model and a combined chart for comparison.
    Metrics are abbreviated as: RSN (Reasoning), MEM (Memory), SCL (Scalability),
    ROB (Robustness), GEN (Generalization), HEAD (Learning Headroom), PAR (Parameter),
    INTP (Interpretability), UNC (Uncertainty), CONT (Continual Learning),
    SPF (Spatial Focus), PAT (Pattern Invariance), FREQ (Frequency Detection),
    RHY (Rhythm Comprehension), TRAJ (Trajectory Prediction).
    
    Args:
        df (pandas.DataFrame): DataFrame containing model scores with metrics as index.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define all 15 metrics with their abbreviations
    metrics_to_plot = {
        'Reasoning_Score': 'RSN',
        'Memory_Score': 'MEM',
        'Scalability_Score': 'SCL',
        'Robustness_Score': 'ROB',
        'Generalization_Score': 'GEN',
        'Learning_Headroom': 'HEAD',
        'Parameter_Score': 'PAR',
        'Interpretability_Score': 'INTP',
        'Uncertainty_Score': 'UNC',
        'Continual_Learning_Score': 'CONT',
        'Spatial_Focus_Score': 'SPF',
        'Pattern_Invariance_Score': 'PAT',
        'Frequency_Detection_Score': 'FREQ',
        'Rhythm_Comprehension_Score': 'RHY',
        'Trajectory_Prediction_Score': 'TRAJ'
    }
    
    df_plot = df.loc[list(metrics_to_plot.keys())].fillna(0)
    labels = list(metrics_to_plot.values())
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # --- Individual Charts ---
    for model_name in df_plot.columns:
        fig_ind, ax_ind = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax_ind.set_ylim(0, 1)
        
        values = df_plot[model_name].values.flatten().tolist()
        values += values[:1]
        
        ax_ind.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax_ind.fill(angles, values, alpha=0.25)
        
        plt.xticks(angles[:-1], labels, color='grey', size=10)
        ax_ind.tick_params(axis='y', labelsize=8)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
        plt.title(f'IQ Profile: {model_name}', size=15, color='black', y=1.1)
        
        filename = f'iq_profile_{model_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        print(f"Individual radar chart saved to '{filename}'")

    # --- Combined Chart ---
    fig_comb, ax_comb = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax_comb.set_ylim(0, 1)
    
    for model_name in df_plot.columns:
        values = df_plot[model_name].values.flatten().tolist()
        values += values[:1]
        ax_comb.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax_comb.fill(angles, values, alpha=0.2)
        
    plt.xticks(angles[:-1], labels, color='grey', size=10)
    ax_comb.tick_params(axis='y', labelsize=8)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
    plt.title('IQ-NET Model Personality Comparison', size=15, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig('iq_net_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_comb)
    print("\nCombined radar chart saved to 'iq_net_radar_comparison.png'")

if __name__ == '__main__':
    print(f"Starting IQ-NET Profiler on device: {PROFILER_CONFIG['device']}")
    models_to_test = {
        "LSTM": LSTMModel, "GRU": GRUModel, "Transformer": TransformerModel,
        "TCN": TCNModel, "CNN": CNNModel, "Zarvan": ZarvanModel
    }
    results = {}

    for name, model_class in models_to_test.items():
        profiler = HolisticAptitudeProfiler(model_class, name, PROFILER_CONFIG)
        profile_scores = profiler.get_profile()
        results[name] = profile_scores
    
    print("\n" + "="*20 + " FINAL IQ-NET EXPANDED REPORT " + "="*20)
    report_df = pd.DataFrame(results).fillna(0)
    report_df.index.name = 'Metric'
    weights = {
        'Memory_Score':20, 'Reasoning_Score':20, 'Scalability_Score':10,
        'Robustness_Score':10, 'Generalization_Score':5, 'Learning_Headroom':5,
        'Parameter_Score':5, 'Continual_Learning_Score':5, 'Interpretability_Score':5,
        'Uncertainty_Score':5, 'Spatial_Focus_Score':10, 'Pattern_Invariance_Score':10,
        'Frequency_Detection_Score':10, 'Rhythm_Comprehension_Score':10,
        'Trajectory_Prediction_Score':10
    }
    for col in weights.keys():
        if col not in report_df.index:
            report_df.loc[col] = 0.0
            
    final_scores = {}
    total_weight = sum(weights.values())
    
    for model_name in report_df.columns:
        model_scores = report_df[model_name]
        weighted_sum = sum(model_scores.get(metric, 0) * w for metric, w in weights.items())
        final_scores[model_name] = (weighted_sum / total_weight) * 100

    report_df.loc['Final_IQ_Score'] = final_scores
    report_df = report_df.reindex(columns=sorted(report_df.columns, key=lambda c: report_df.loc['Final_IQ_Score', c], reverse=True))
    
    print(report_df.round(4).to_markdown())
    
    plot_radar_charts(report_df)