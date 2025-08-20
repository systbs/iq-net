# ============================================================================
#
#       IQ-NET: Holistic Aptitude Profiler (Version 1)
#
# This is the final refactored version of the code for benchmarking neural network models
# and generating radar charts for their performance across 15 metrics.
# It is designed for clarity, reproducibility, modularity, and GitHub presentation.
# The radar charts visualize each model's profile using abbreviations:
# RSN (Reasoning), MEM (Memory), SCL (Scalability), ROB (Robustness), GEN (Generalization),
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
from scipy import signal

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

# --- BENCHMARK CONFIGURATION ---
# This config holds parameters for the benchmark itself, data generation, and probe tasks.
# It is completely independent of any model's internal structure.
BENCHMARK_CONFIG = {
    'vocab_size': 100,
    'sequence_length': 128,
    'num_samples': 1200,
    'batch_size': 32,
    'probe_training_steps': 700,
    'learning_rate': 1e-3,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'image_size': 32,
    'audio_signal_length': 2048,
    'video_num_frames': 16,
    'num_classes': 10, # Number of classes for probe classification tasks
}

# --- MODEL CONFIGURATIONS ---
# Each model's specific configuration is defined here.
# These configs are passed directly to the model's constructor.
# `vocab_size` and `num_classes` are injected by the profiler from BENCHMARK_CONFIG.
MODEL_CONFIGS = {
    'Zarvan': {
        'embed_dim': 128,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
    },
    'LSTM': {
        'embed_dim': 128,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'GRU': {
        'embed_dim': 128,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'Transformer': { # Simple Encoder-only Transformer
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'ffn_dim': 128,
        'dropout': 0.1,
    },
    'CNN_1D': { # Simple 1D CNN for sequences
        'embed_dim': 128, # Embedding dim for input tokens
        'num_channels': [64, 128], # Channels for conv layers
        'kernel_sizes': [3, 3],   # Kernel sizes for conv layers
        'dropout': 0.2,
    },
    # Example for another model (e.g., a simple RNN)
    # 'SimpleRNN': {
    #     'hidden_dim': 128,
    #     'num_layers': 2,
    #     'dropout': 0.2,
    # },
}

# ===========================================
# 1. Model Implementations (Example: Zarvan)
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

# --- Holistic ---
class _HolisticExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.filter = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=2)
        head_outputs = torch.sum(weights * v, dim=2, keepdim=True)
        return self.filter(head_outputs.reshape(B, 1, E))

# --- Associative ---
class _AssociativeExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj = nn.Linear(embed_dim, 1)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        return torch.sum(weights * v, dim=1, keepdim=True)

class _SequentialExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.angle_calculator = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        weights = torch.cumsum(s * v, dim=1)
        alpha = self.norm(self.angle_calculator(weights / S))
        omega = alpha * math.pi
        phases = torch.cat([torch.cos(omega), torch.sin(omega)], dim=-1)
        return self.out_proj(phases)

    
# --- Zarvan Block ---
class _ZarvanBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        
        self.input_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        self.holistic_ctx = _HolisticExtractor(embed_dim, num_heads)
        self.associative_ctx = _AssociativeExtractor(embed_dim)
        self.sequential_ctx = _SequentialExtractor(embed_dim)
        
        self.expert_gate = nn.Sequential(
            nn.Linear(embed_dim, 3),
            nn.SiLU()
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        x_residual = x
        x_adapted = self.input_adapter(x)
        
        q_holistic = self.holistic_ctx(x_adapted)
        q_associative = self.associative_ctx(x_adapted)
        q_sequential = self.sequential_ctx(x_adapted)

        gates = self.expert_gate(x_adapted)
        g_h, g_a, g_s = gates.chunk(3, dim=-1)
        h_candidate = (
            g_h * q_holistic.expand(-1, S, -1) +
            g_a * q_associative.expand(-1, S, -1) +
            g_s * q_sequential
        )
        
        out = x_residual + self.ffn(self.norm(h_candidate))
        return out

class ZarvanModel(nn.Module):
    # --- MODEL INTERFACE CONTRACT ---
    # All models must define an `output_feature_dim` attribute.
    # This allows the profiler to be agnostic to the model type.
    def __init__(self, config):
        super().__init__()
        # Inject vocab_size and num_classes from benchmark config
        vocab_size = config['vocab_size']
        num_classes = config['num_classes']

        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim'])
        self.layers = nn.ModuleList([
            _ZarvanBlock(config['embed_dim'], config['hidden_dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ])
        self.output_head = nn.Linear(config['embed_dim'], num_classes)

        # --- MODEL INTERFACE CONTRACT ---
        # Expose the feature dimension for probe heads
        self.output_feature_dim = config['embed_dim']

    def forward(self, x, embeds=None, return_features=False):
        h = self.pos_encoder(self.embedding(x) if embeds is None else embeds)
        for layer in self.layers:
            h = layer(h)
        if return_features:
            return h
        return self.output_head(h)

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        num_classes = config['num_classes']
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.lstm = nn.LSTM(config['embed_dim'], config['hidden_dim'], config['num_layers'], 
                            batch_first=True, dropout=config['dropout'] if config['num_layers'] > 1 else 0)

        # --- START: FIX ---
        # Add a projection layer if embed_dim and hidden_dim are different
        # This handles inputs from perception heads which use hidden_dim
        if config['embed_dim'] != config['hidden_dim']:
            self.input_proj = nn.Linear(config['hidden_dim'], config['embed_dim'])
        else:
            self.input_proj = nn.Identity()
        # --- END: FIX ---

        self.output_head = nn.Linear(config['hidden_dim'], num_classes)
        self.output_feature_dim = config['hidden_dim'] 

    def forward(self, x, embeds=None, return_features=False):
        if embeds is None:
            h = self.embedding(x)  # [B, S, embed_dim]
        else:
            # Project the embeddings from perception heads (sized hidden_dim) 
            # to the size expected by the LSTM layer (embed_dim).
            h = self.input_proj(embeds) # [B, S, hidden_dim] -> [B, S, embed_dim]
        
        lstm_out, (hidden, cell) = self.lstm(h)
        
        if return_features:
            return lstm_out
        else:
            logits = self.output_head(lstm_out)
            return logits

# --- GRU Model ---
class GRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        num_classes = config['num_classes']
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.gru = nn.GRU(config['embed_dim'], config['hidden_dim'], config['num_layers'], 
                          batch_first=True, dropout=config['dropout'] if config['num_layers'] > 1 else 0)
        
        # --- START: FIX ---
        # Add a projection layer if embed_dim and hidden_dim are different
        if config['embed_dim'] != config['hidden_dim']:
            self.input_proj = nn.Linear(config['hidden_dim'], config['embed_dim'])
        else:
            self.input_proj = nn.Identity()
        # --- END: FIX ---

        self.output_head = nn.Linear(config['hidden_dim'], num_classes)
        self.output_feature_dim = config['hidden_dim']

    def forward(self, x, embeds=None, return_features=False):
        if embeds is None:
            h = self.embedding(x)
        else:
            # Project the embeddings from perception heads
            h = self.input_proj(embeds)

        gru_out, hidden = self.gru(h)
        
        if return_features:
            return gru_out # [B, S, H]
        else:
            logits = self.output_head(gru_out)
            return logits

# --- Simple Transformer Model ---
class SimpleTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        num_classes = config['num_classes']
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.pos_encoder = PositionalEncoding(config['embed_dim']) # Reuse from Zarvan
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embed_dim'], nhead=config['num_heads'],
            dim_feedforward=config['ffn_dim'], dropout=config['dropout'],
            activation='relu', batch_first=True # Important for newer PyTorch versions
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        self.output_head = nn.Linear(config['embed_dim'], num_classes)
        # Interface contract
        self.output_feature_dim = config['embed_dim']

    def forward(self, x, embeds=None, return_features=False):
        h = self.pos_encoder(self.embedding(x) if embeds is None else embeds) # [B, S, E]
        features = self.transformer_encoder(h) # [B, S, E]
        
        if return_features:
            return features # [B, S, E]
        else:
            # Apply the head to every token in the sequence, DO NOT POOL
            return self.output_head(features) # [B, S, C]

# --- Simple 1D CNN Model ---
class CNN1DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config['vocab_size']
        num_classes = config['num_classes']
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        
        # --- START: FIX ---
        # Add a 1x1 conv to project inputs from perception heads
        # from output_feature_dim to the model's internal embed_dim
        if config['num_channels'][-1] != config['embed_dim']:
            self.input_proj = nn.Conv1d(config['num_channels'][-1], config['embed_dim'], kernel_size=1)
        else:
            self.input_proj = nn.Identity()
        # --- END: FIX ---

        layers = []
        in_channels = config['embed_dim']
        for out_channels, kernel_size in zip(config['num_channels'], config['kernel_sizes']):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            if config['dropout'] > 0:
                layers.append(nn.Dropout(config['dropout']))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(config['num_channels'][-1], num_classes)
        self.output_feature_dim = config['num_channels'][-1]

    def forward(self, x, embeds=None, return_features=False):
        if embeds is None:
            h = self.embedding(x)  # [B, S, embed_dim]
            h = h.transpose(1, 2)  # [B, embed_dim, S]
        else:
            # 'embeds' comes from perception heads with size output_feature_dim
            h = embeds.transpose(1, 2)      # [B, output_feature_dim, S]
            h = self.input_proj(h)          # Project to [B, embed_dim, S]

        conv_out = self.conv_layers(h) # [B, C_out, S]
        features = conv_out.transpose(1, 2) # [B, S, C_out]

        if return_features:
            return features
        else:
            return self.output_head(features)

class SpatialProcessor(nn.Module):
    """
    A lightweight Transformer-based module to process the patch sequence of a single frame.
    It takes a sequence of patches [B, S, E] and outputs a single feature vector [B, E]
    by pooling the information from the [CLS] token, making it a rich summary of the frame.
    """
    def __init__(self, embed_dim, num_heads=4, ffn_dim=128, dropout=0.1):
        super().__init__()
        # A single, powerful Transformer encoder layer to process spatial patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, S, E] (a batch of patch sequences from frames)
        features = self.transformer_encoder(x)
        # We only return the feature of the [CLS] token, which is the first token
        cls_feature = features[:, 0, :]
        return self.norm(cls_feature)
    
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
                    seq[reas_start: reas_start + reas_len] = PAD
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

class ImageAsSequenceHead(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        patches = x.unfold(2, P, P).unfold(3, P, P).contiguous()
        patches = patches.view(B, C, self.num_patches, P, P)
        patches = patches.permute(0, 2, 3, 4, 1).reshape(B, self.num_patches, -1)
        embedded_patches = self.patch_embedding(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, embedded_patches), dim=1)
        x_seq += self.pos_embedding
        return x_seq

class AudioAsSequenceHead(nn.Module):
    def __init__(self, signal_length=2048, frame_size=256, embed_dim=64):
        super().__init__()
        assert signal_length % frame_size == 0, "Signal length must be divisible by frame size."
        self.frame_size = frame_size
        self.num_frames = signal_length // frame_size
        self.frame_embedding = nn.Linear(frame_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        frames = x.unfold(2, self.frame_size, self.frame_size).contiguous()
        frames = frames.permute(0, 2, 1, 3).reshape(B, self.num_frames, -1)
        embedded_frames = self.frame_embedding(frames)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, embedded_frames), dim=1)
        x_seq += self.pos_embedding
        return x_seq

# --- Video Sequence Model ---
# A simple model to aggregate frame features for trajectory prediction
class VideoSequenceModel(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, E]
        x = self.transformer(x)
        # Use mean pooling of [CLS] tokens across frames
        x = x.mean(dim=1) # [B, E]
        return self.norm(x)

class HolisticAptitudeProfiler:
    """
    A profiler for benchmarking neural network models on a diverse set of cognitive tasks.
    This class is completely independent of the model's internal structure.
    It interacts with models only through a defined interface (e.g., output_feature_dim).
    """
    def __init__(self, model_class, model_name, benchmark_config, model_config):
        self.model_class = model_class
        self.model_name = model_name
        self.benchmark_config = benchmark_config # Corrected: Store benchmark config
        self.model_config = model_config         # Corrected: Store model config
        self.device = benchmark_config['device']
        set_seed(42)
        print(f"--- Profiling {self.model_name} ---")

        # --- MODEL CONFIGURATION INJECTION ---
        # Inject benchmark-wide config into the model-specific config
        full_model_config = {**model_config, **benchmark_config}

        # --- MODEL INSTANTIATION ---
        # Create a temporary model instance to access its interface contract
        temp_model = self.model_class(full_model_config)
        
        # --- PROBE HEAD CONFIGURATION ---
        # Use the model's declared output feature dimension for probe heads
        feature_dim = temp_model.output_feature_dim

        self.image_perception_head = ImageAsSequenceHead(
            img_size=self.benchmark_config['image_size'],
            patch_size=8,
            embed_dim=feature_dim # Use model's feature dim
        ).to(self.device)
        self.audio_perception_head = AudioAsSequenceHead(
            signal_length=self.benchmark_config['audio_signal_length'],
            frame_size=256,
            embed_dim=feature_dim # Use model's feature dim
        ).to(self.device)
        # Video sequence model to process frame features
        self.video_sequence_model = VideoSequenceModel(embed_dim=feature_dim).to(self.device)

        self.regression_head = nn.Linear(feature_dim, 2).to(self.device)
        self.multidomain_classification_head = nn.Linear(feature_dim, self.benchmark_config['num_classes']).to(self.device)

    def get_profile(self):
        profile = {}
        probe_data = generate_composite_probe_data(
            self.benchmark_config['num_samples'], self.benchmark_config['sequence_length'],
            self.benchmark_config['vocab_size'], self.benchmark_config['num_classes']
        )
        print("Running static tests...")
        profile['Parameter_Score'] = self._compute_parameter_score()
        print(f"  > Parameter_Score calculated: {profile['Parameter_Score']:.4f}")
        profile['Scalability_Score'] = self._compute_scalability_score()
        print(f"  > Scalability_Score calculated: {profile['Scalability_Score']:.4f}")
        print("Running probe training and evaluation...")
        # --- MODEL CONFIGURATION INJECTION (again for final model) ---
        full_model_config = {**self.model_config, **self.benchmark_config}
        model = self.model_class(full_model_config).to(self.device)
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
        coords = np.zeros((batch_size, 2), dtype=np.float32)
        shape_color = np.array([0.9, 0.1, 0.1])
        for i in range(batch_size):
            shape_size = img_size // 4 if level == 1 else img_size // 8
            if level == 2:
                try:
                    for _ in range(5):
                        x1, y1, x2, y2 = np.random.randint(0, img_size, 4)
                        cv2.line(images[i], (x1, y1), (x2, y2), np.random.rand(3), 1)
                except:
                    pass # Handle if cv2 is not available
            elif level == 3:
                try:
                    for _ in range(5):
                        dx, dy = np.random.randint(shape_size, img_size - shape_size, 2)
                        d_size = shape_size // 2
                        images[i, dy - d_size:dy + d_size, dx - d_size:dx + d_size] = shape_color
                except:
                    pass
            # Ensure shape is fully within bounds
            cx, cy = np.random.randint(shape_size, img_size - shape_size, 2)
            start_y, end_y = max(0, cy - shape_size), min(img_size, cy + shape_size)
            start_x, end_x = max(0, cx - shape_size), min(img_size, cx + shape_size)
            images[i, start_y:end_y, start_x:end_x] = shape_color
            coords[i] = [cx / img_size, cy / img_size]
        return torch.from_numpy(images).permute(0, 3, 1, 2), torch.from_numpy(coords)

    def _compute_spatial_focus_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Spatial Focus Score...")
        for level, name in levels.items():
            # --- 1. Generate Data ---
            images, true_coords = self._generate_spatial_focus_data(level, self.benchmark_config['batch_size'] * 4, self.benchmark_config['image_size']) # More data for training
            images, true_coords = images.to(self.device), true_coords.to(self.device)
            
            # --- 2. Create a temporary model instance and perception head for training ---
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()  # Set to train mode for cuDNN backward pass
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_image_head = copy.deepcopy(self.image_perception_head).to(self.device)
            temp_regression_head = copy.deepcopy(self.regression_head).to(self.device)
            
            # --- 3. Define optimizer for the perception and probe heads only ---
            optimizer = optim.AdamW(list(temp_image_head.parameters()) + list(temp_regression_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1) # Lower LR often helps
            
            # --- 4. Training Loop for this specific probe ---
            temp_image_head.train()
            temp_regression_head.train()
            num_training_steps = 1000 # Increased steps
            loss_fn = nn.MSELoss() # Use MSE for regression
            for step in range(num_training_steps):
                 # Sample a batch
                 batch_indices = torch.randperm(images.size(0))[:self.benchmark_config['batch_size']]
                 batch_images = images[batch_indices]
                 batch_coords = true_coords[batch_indices]
                 
                 optimizer.zero_grad()
                 embeddings = temp_image_head(batch_images)
                 model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                 cls_output = model_features.mean(dim=1)
                 predicted_coords = torch.sigmoid(temp_regression_head(cls_output))
                 loss = loss_fn(predicted_coords, batch_coords)
                 if not torch.isnan(loss):
                     loss.backward()
                     optimizer.step()

            # --- 5. Evaluation Loop (using the trained temporary heads) ---
            temp_image_head.eval()
            temp_regression_head.eval()
            eval_losses = []
            with torch.no_grad():
                 dataset_eval = TensorDataset(images, true_coords)
                 loader_eval = DataLoader(dataset_eval, batch_size=self.benchmark_config['batch_size'])
                 for eval_images, eval_coords in loader_eval:
                     eval_images, eval_coords = eval_images.to(self.device), eval_coords.to(self.device)
                     embeddings = temp_image_head(eval_images)
                     model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                     cls_output = model_features.mean(dim=1)
                     predicted_coords = torch.sigmoid(temp_regression_head(cls_output))
                     # Use L1 distance for scoring, less sensitive than L2 squared
                     distance = torch.abs(predicted_coords - eval_coords).sum(dim=1) # [B]
                     eval_losses.extend(distance.cpu().numpy())

            # --- 6. Calculate final score for this level ---
            if eval_losses:
                mean_distance = np.mean(eval_losses)
                # Use a less sensitive exponential for scoring
                score = math.exp(-2.0 * mean_distance) # Changed from -10 to -2
            else:
                score = 0.0
            level_scores[level] = score
            print(f"    - Level {level} ({name}) Score: {score:.4f}")

        # --- 7. Calculate weighted final score ---
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Spatial_Focus_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _generate_pattern_invariance_data(self, level, batch_size=32, img_size=32):
        images, labels = np.zeros((batch_size, img_size, img_size, 1), dtype=np.float32), np.zeros(batch_size, dtype=int)
        pattern = np.zeros((7, 7))
        pattern[1:6, 1] = 1
        pattern[5, 1:4] = 1
        for i in range(batch_size):
            img = np.zeros((img_size, img_size))
            is_target = random.random() > 0.5
            current_pattern = pattern if is_target else np.rot90(pattern, k=2)
            p_size_h, p_size_w = 7, 7
            if level >= 2:
                new_size = random.randint(5, 12)
                current_pattern = cv2.resize(current_pattern, (new_size, new_size)) if 'cv2' in globals() else current_pattern[:new_size, :new_size]
                p_size_h, p_size_w = current_pattern.shape
            if level >= 3:
                angle = random.uniform(-45, 45)
                current_pattern = rotate(current_pattern, angle, reshape=True, mode='constant', cval=0.0)
                p_size_h, p_size_w = current_pattern.shape
            # Ensure pattern is fully within bounds
            if p_size_h >= img_size or p_size_w >= img_size:
                images[i, :, :, 0] = np.zeros((img_size, img_size))
                labels[i] = 0
                continue
            x, y = np.random.randint(0, img_size - p_size_w), np.random.randint(0, img_size - p_size_h)
            img[y:y + p_size_h, x:x + p_size_w] = current_pattern
            images[i, :, :, 0] = img
            labels[i] = 1 if is_target else 0
        return torch.from_numpy(images).permute(0, 3, 1, 2).repeat(1, 3, 1, 1), torch.from_numpy(labels)

    def _compute_pattern_invariance_score(self, model):
        levels, weights = {1: 'translation', 2: 'scale', 3: 'rotation'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Pattern Invariance Score...")
        for level, name in levels.items():
            # --- 1. Generate Data ---
            images, labels = self._generate_pattern_invariance_data(level, self.benchmark_config['batch_size'] * 4, self.benchmark_config['image_size'])
            
            # --- 2. Create temporary models and heads ---
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()  # Set to train mode for cuDNN backward pass
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_image_head = copy.deepcopy(self.image_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            
            # --- 3. Define optimizer ---
            optimizer = optim.AdamW(list(temp_image_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            
            # --- 4. Training Loop ---
            temp_image_head.train()
            temp_classification_head.train()
            num_training_steps = 1000 # Increased steps
            dataset_train = TensorDataset(images, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            
            for step in range(num_training_steps):
                try:
                    batch_images, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_images, batch_labels = next(iter(loader_train))
                
                batch_images, batch_labels = batch_images.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                
                embeddings = temp_image_head(batch_images)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

            # --- 5. Evaluation Loop ---
            temp_image_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_images, eval_labels in loader_eval:
                    eval_images, eval_labels = eval_images.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_image_head(eval_images)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
            
        # --- 6. Calculate weighted final score ---
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Pattern_Invariance_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _compute_audio_profile(self, model):
        audio_scores = {}
        audio_scores['Frequency_Detection_Score'] = self._compute_frequency_detection_score(model)
        audio_scores['Rhythm_Comprehension_Score'] = self._compute_rhythm_comprehension_score(model)
        audio_scores['Texture_Detection_Score'] = self._compute_texture_detection_score(model)
        audio_scores['Energy_Envelope_Score'] = self._compute_energy_envelope_score(model)
        audio_scores['Prosody_Detection_Score'] = self._compute_prosody_detection_score(model)
        audio_scores['Formant_Dynamics_Score'] = self._compute_formant_dynamics_score(model)
        return audio_scores

    def _generate_frequency_detection_data(self, level, batch_size=32, sr=8000, length=2048):
        signals, labels = np.zeros((batch_size, length), dtype=np.float32), np.zeros(batch_size, dtype=int)
        freqs = [261, 329, 392, 440] # C4, E4, G4, A4
        t = np.linspace(0, length / sr, length, endpoint=False)
        for i in range(batch_size):
            target_freq_idx = random.randint(0, len(freqs) - 1)
            target_freq = freqs[target_freq_idx]
            amp, noise_amp = (0.8, 0.2) if level == 1 else ((0.2, 0.2) if level == 2 else (0.5, 0.2))
            signal = np.random.randn(length) * noise_amp
            signal += np.sin(2 * np.pi * target_freq * t) * amp
            if level == 3:
                distractor_freq = random.choice([f for f in freqs if f != target_freq])
                signal += np.sin(2 * np.pi * distractor_freq * t) * (amp * 0.8)
            signals[i, :] = signal
            labels[i] = target_freq_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_frequency_detection_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Frequency Detection Score...")
        for level, name in levels.items():
            # --- 1. Generate Data ---
            signals, labels = self._generate_frequency_detection_data(level, self.benchmark_config['batch_size'] * 4)
            
            # --- 2. Create temporary models and heads ---
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()  # Set to train mode for cuDNN backward pass
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_audio_head = copy.deepcopy(self.audio_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            
            # --- 3. Define optimizer ---
            optimizer = optim.AdamW(list(temp_audio_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            
            # --- 4. Training Loop ---
            temp_audio_head.train()
            temp_classification_head.train()
            num_training_steps = 400 # Increased steps
            dataset_train = TensorDataset(signals, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            
            for step in range(num_training_steps):
                try:
                    batch_signals, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_signals, batch_labels = next(iter(loader_train))
                
                batch_signals, batch_labels = batch_signals.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                
                embeddings = temp_audio_head(batch_signals)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

            # --- 5. Evaluation Loop ---
            temp_audio_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_signals, eval_labels in loader_eval:
                    eval_signals, eval_labels = eval_signals.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_audio_head(eval_signals)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
            
        # --- 6. Calculate weighted final score ---
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Frequency_Detection_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _generate_rhythm_comprehension_data(self, level, batch_size=32, length=2048):
        signals = np.zeros((batch_size, length), dtype=np.float32)
        labels = np.zeros(batch_size, dtype=int)
        patterns = [[1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]]
        for i in range(batch_size):
            pattern_idx = random.randint(0, len(patterns) - 1)
            pattern = patterns[pattern_idx]
            if level >= 2:
                pattern = pattern + random.choice(patterns)
            pulse_len = 100
            signal = np.zeros(length)
            pos = 50
            for beat in pattern:
                jitter = int(random.uniform(-20, 20)) if level == 3 else 0
                if beat == 1 and pos + pulse_len + jitter < length and pos + jitter >= 0:
                    signal[pos + jitter:pos + jitter + pulse_len] = np.sin(np.linspace(0, 3 * np.pi, pulse_len))
                pos += pulse_len * 2
            signals[i, :] = signal + np.random.randn(length) * 0.1
            labels[i] = pattern_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_rhythm_comprehension_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Rhythm Comprehension Score...")
        for level, name in levels.items():
           # --- 1. Generate Data ---
            signals, labels = self._generate_rhythm_comprehension_data(level, self.benchmark_config['batch_size'] * 4)
            
            # --- 2. Create temporary models and heads ---
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()  # Set to train mode for cuDNN backward pass
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_audio_head = copy.deepcopy(self.audio_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            
            # --- 3. Define optimizer ---
            optimizer = optim.AdamW(list(temp_audio_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            
            # --- 4. Training Loop ---
            temp_audio_head.train()
            temp_classification_head.train()
            num_training_steps = 400 # Increased steps
            dataset_train = TensorDataset(signals, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            
            for step in range(num_training_steps):
                try:
                    batch_signals, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_signals, batch_labels = next(iter(loader_train))
                
                batch_signals, batch_labels = batch_signals.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                
                embeddings = temp_audio_head(batch_signals)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()

            # --- 5. Evaluation Loop ---
            temp_audio_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_signals, eval_labels in loader_eval:
                    eval_signals, eval_labels = eval_signals.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_audio_head(eval_signals)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
            
        # --- 6. Calculate weighted final score ---
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Rhythm_Comprehension_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _generate_texture_detection_data(self, level, batch_size=32, sr=8000, length=2048):
        signals, labels = np.zeros((batch_size, length), dtype=np.float32), np.zeros(batch_size, dtype=int)
        t = np.linspace(0, length / sr, length, endpoint=False)
        # فرم‌های مصنوعی (شبیه به formantهای گفتار)
        formant_sets = [
            [200, 800, 2500],  # شبیه صدای "a"
            [400, 1200, 2800], # شبیه صدای "e"
            [600, 1000, 3000]  # شبیه صدای "o"
        ]
        for i in range(batch_size):
            formant_idx = random.randint(0, len(formant_sets) - 1)
            formants = formant_sets[formant_idx]
            amp = 0.8 if level == 1 else (0.5 if level == 2 else 0.3)
            noise_amp = 0.1 if level == 1 else (0.2 if level == 2 else 0.3)
            signal = np.random.randn(length) * noise_amp
            # ترکیب چند فرکانس (شبیه formant)
            for freq in formants:
                signal += np.sin(2 * np.pi * freq * t) * (amp / len(formants))
            # اضافه کردن AM برای سطح سخت
            if level == 3:
                am_freq = random.uniform(2, 5)  # فرکانس AM بین 2-5 Hz
                signal *= (1 + 0.5 * np.sin(2 * np.pi * am_freq * t))
            signals[i, :] = signal
            labels[i] = formant_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_texture_detection_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Texture Detection Score...")
        for level, name in levels.items():
            signals, labels = self._generate_texture_detection_data(level, self.benchmark_config['batch_size'] * 4)
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_audio_head = copy.deepcopy(self.audio_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            optimizer = optim.AdamW(list(temp_audio_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            temp_audio_head.train()
            temp_classification_head.train()
            num_training_steps = 400
            dataset_train = TensorDataset(signals, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            for step in range(num_training_steps):
                try:
                    batch_signals, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_signals, batch_labels = next(iter(loader_train))
                batch_signals, batch_labels = batch_signals.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                embeddings = temp_audio_head(batch_signals)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
            temp_audio_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_signals, eval_labels in loader_eval:
                    eval_signals, eval_labels = eval_signals.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_audio_head(eval_signals)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Texture_Detection_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _generate_energy_envelope_data(self, level, batch_size=32, sr=8000, length=2048):
        signals, labels = np.zeros((batch_size, length), dtype=np.float32), np.zeros(batch_size, dtype=int)
        t = np.linspace(0, length / sr, length, endpoint=False)
        envelope_patterns = [
            [1, 0.5, 1, 0.5],  # الگوی متناوب
            [1, 1, 0.5, 0.5],  # الگوی نزولی
            [0.5, 1, 1, 0.5]   # الگوی میانی
        ]
        for i in range(batch_size):
            pattern_idx = random.randint(0, len(envelope_patterns) - 1)
            pattern = envelope_patterns[pattern_idx]
            signal = np.sin(2 * np.pi * 440 * t)  # فرکانس پایه (A4)
            pos = 0
            step = length // len(pattern)
            for amp in pattern:
                if level == 3:
                    amp *= random.uniform(0.8, 1.2)  # اضافه کردن نویز به شدت
                signal[pos:pos + step] *= amp
                pos += step
            signal += np.random.randn(length) * (0.1 if level == 1 else 0.2 if level == 2 else 0.3)
            signals[i, :] = signal
            labels[i] = pattern_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_energy_envelope_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Energy Envelope Detection Score...")
        for level, name in levels.items():
            signals, labels = self._generate_energy_envelope_data(level, self.benchmark_config['batch_size'] * 4)
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_audio_head = copy.deepcopy(self.audio_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            optimizer = optim.AdamW(list(temp_audio_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            temp_audio_head.train()
            temp_classification_head.train()
            num_training_steps = 400
            dataset_train = TensorDataset(signals, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            for step in range(num_training_steps):
                try:
                    batch_signals, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_signals, batch_labels = next(iter(loader_train))
                batch_signals, batch_labels = batch_signals.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                embeddings = temp_audio_head(batch_signals)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
            temp_audio_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_signals, eval_labels in loader_eval:
                    eval_signals, eval_labels = eval_signals.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_audio_head(eval_signals)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Energy_Envelope_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _generate_prosody_detection_data(self, level, batch_size=32, sr=8000, length=2048):
        signals, labels = np.zeros((batch_size, length), dtype=np.float32), np.zeros(batch_size, dtype=int)
        t = np.linspace(0, length / sr, length, endpoint=False)
        prosody_patterns = [
            [440, 440, 494, 494],  # ثابت سپس افزایش
            [494, 494, 440, 440],  # ثابت سپس کاهش
            [440, 494, 494, 440]   # افزایش سپس کاهش
        ]
        for i in range(batch_size):
            pattern_idx = random.randint(0, len(prosody_patterns) - 1)
            pattern = prosody_patterns[pattern_idx]
            signal = np.zeros(length)
            step = length // len(pattern)
            for j, freq in enumerate(pattern):
                if level == 3:
                    freq *= random.uniform(0.9, 1.1)  # نویز در فرکانس
                signal[j * step:(j + 1) * step] = np.sin(2 * np.pi * freq * t[j * step:(j + 1) * step])
            signal += np.random.randn(length) * (0.1 if level == 1 else 0.2 if level == 2 else 0.3)
            signals[i, :] = signal
            labels[i] = pattern_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_prosody_detection_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Prosody Detection Score...")
        for level, name in levels.items():
            signals, labels = self._generate_prosody_detection_data(level, self.benchmark_config['batch_size'] * 4)
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_audio_head = copy.deepcopy(self.audio_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            optimizer = optim.AdamW(list(temp_audio_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            temp_audio_head.train()
            temp_classification_head.train()
            num_training_steps = 400
            dataset_train = TensorDataset(signals, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            for step in range(num_training_steps):
                try:
                    batch_signals, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_signals, batch_labels = next(iter(loader_train))
                batch_signals, batch_labels = batch_signals.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                embeddings = temp_audio_head(batch_signals)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
            temp_audio_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_signals, eval_labels in loader_eval:
                    eval_signals, eval_labels = eval_signals.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_audio_head(eval_signals)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Prosody_Detection_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _generate_formant_dynamics_data(self, level, batch_size=32, sr=8000, length=2048):
        signals, labels = np.zeros((batch_size, length), dtype=np.float32), np.zeros(batch_size, dtype=int)
        t = np.linspace(0, length / sr, length, endpoint=False)
        formant_patterns = [
            [(200, 800), (800, 2500)],  # glide از "a" به "i"
            [(400, 1200), (600, 1000)], # glide از "e" به "o"
            [(600, 1000), (200, 800)]   # glide از "o" به "a"
        ]
        for i in range(batch_size):
            pattern_idx = random.randint(0, len(formant_patterns) - 1)
            pattern = formant_patterns[pattern_idx]
            signal = np.zeros(length)
            step = length // 2
            for j, (f1, f2) in enumerate(pattern):
                start_freq1, end_freq1 = f1, pattern[(j + 1) % 2][0] if level >= 2 else f1
                start_freq2, end_freq2 = f2, pattern[(j + 1) % 2][1] if level >= 2 else f2
                freq1 = np.linspace(start_freq1, end_freq1, step)
                freq2 = np.linspace(start_freq2, end_freq2, step)
                signal[j * step:(j + 1) * step] = (
                    np.sin(2 * np.pi * freq1 * t[:step]) +
                    np.sin(2 * np.pi * freq2 * t[:step])
                ) / 2
            signal += np.random.randn(length) * (0.1 if level == 1 else 0.2 if level == 2 else 0.3)
            if level == 3:
                signal *= (1 + 0.5 * np.sin(2 * np.pi * random.uniform(2, 5) * t))  # AM تصادفی
            signals[i, :] = signal
            labels[i] = pattern_idx
        return torch.from_numpy(signals), torch.from_numpy(labels)

    def _compute_formant_dynamics_score(self, model):
        levels, weights = {1: 'easy', 2: 'medium', 3: 'hard'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Formant Dynamics Detection Score...")
        for level, name in levels.items():
            signals, labels = self._generate_formant_dynamics_data(level, self.benchmark_config['batch_size'] * 4)
            temp_model = copy.deepcopy(model).to(self.device)
            temp_model.train()
            for param in temp_model.parameters():
                param.requires_grad = False
            temp_audio_head = copy.deepcopy(self.audio_perception_head).to(self.device)
            temp_classification_head = copy.deepcopy(self.multidomain_classification_head).to(self.device)
            optimizer = optim.AdamW(list(temp_audio_head.parameters()) + list(temp_classification_head.parameters()), lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.CrossEntropyLoss()
            temp_audio_head.train()
            temp_classification_head.train()
            num_training_steps = 400
            dataset_train = TensorDataset(signals, labels)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
            for step in range(num_training_steps):
                try:
                    batch_signals, batch_labels = next(iter(loader_train))
                except StopIteration:
                    loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'], shuffle=True)
                    batch_signals, batch_labels = next(iter(loader_train))
                batch_signals, batch_labels = batch_signals.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                embeddings = temp_audio_head(batch_signals)
                model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                cls_output = model_features.mean(dim=1)
                logits = temp_classification_head(cls_output)
                loss = loss_fn(logits, batch_labels)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
            temp_audio_head.eval()
            temp_classification_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'])
                for eval_signals, eval_labels in loader_eval:
                    eval_signals, eval_labels = eval_signals.to(self.device), eval_labels.to(self.device)
                    embeddings = temp_audio_head(eval_signals)
                    model_features = temp_model(x=None, embeds=embeddings, return_features=True)
                    cls_output = model_features.mean(dim=1)
                    logits = temp_classification_head(cls_output)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == eval_labels).sum().item()
                    total += eval_labels.size(0)
            accuracy = correct / total if total > 0 else 0
            level_scores[level] = accuracy
            print(f"    - Level {level} ({name}) Accuracy: {accuracy:.4f}")
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Formant_Dynamics_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _compute_video_profile(self, model):
        video_scores = {}
        video_scores['Trajectory_Prediction_Score'] = self._compute_trajectory_prediction_score(model)
        return video_scores

    def _generate_trajectory_prediction_data(self, level, batch_size=32, num_frames=16, img_size=32):
        videos = np.zeros((batch_size, num_frames, img_size, img_size, 3), dtype=np.float32)
        final_coords = np.zeros((batch_size, 2), dtype=np.float32)
        shape_size = img_size // 8
        for i in range(batch_size):
            start_x, start_y = np.random.randint(shape_size, img_size // 2, 2)
            if level == 1:
                vx, vy, gx, gy = *np.random.uniform(0.5, 2.0, 2), 0, 0
            elif level == 2:
                vx, vy, gx, gy = np.random.uniform(2.0, 3.0), np.random.uniform(-1.0, -0.5), 0, 0.1
            else:
                vx, vy, gx, gy = *np.random.uniform(1.5, 2.5, 2), 0, 0
            x, y = float(start_x), float(start_y)
            for t in range(num_frames):
                x, y = x + vx, y + vy
                vx, vy = vx + gx, vy + gy
                if level == 3:
                    if not (shape_size < x < img_size - shape_size):
                        vx *= -1
                    if not (shape_size < y < img_size - shape_size):
                        vy *= -1
                if t < 10: # Draw object for first 10 frames
                    ix, iy = int(x), int(y)
                    if 0 <= ix < img_size and 0 <= iy < img_size:
                        start_y_c, end_y_c = max(0, iy - shape_size), min(img_size, iy + shape_size)
                        start_x_c, end_x_c = max(0, ix - shape_size), min(img_size, ix + shape_size)
                        videos[i, t, start_y_c:end_y_c, start_x_c:end_x_c] = [0.1, 0.9, 0.1]
            final_coords[i] = np.clip([x / img_size, y / img_size], 0, 1)
        return torch.from_numpy(videos).permute(0, 1, 4, 2, 3), torch.from_numpy(final_coords)

    def _compute_trajectory_prediction_score(self, model):
        levels, weights = {1: 'linear', 2: 'parabolic', 3: 'bounce'}, {1: 1, 2: 2, 3: 4}
        level_scores = {}
        print("  > Calculating Trajectory Prediction Score (Two-Stage Method)...")
        for level, name in levels.items():
            # --- 1. Generate Data ---
            num_train_samples = self.benchmark_config['batch_size'] * 8
            videos, true_coords = self._generate_trajectory_prediction_data(
                level, num_train_samples, self.benchmark_config['video_num_frames'], self.benchmark_config['image_size']
            )
            videos, true_coords = videos.to(self.device), true_coords.to(self.device)
            B, T, C, H, W = videos.shape

            # --- 2. Create temporary models and heads ---
            temp_model = self.model_class({**self.model_config, **self.benchmark_config}).to(self.device)
            temp_model.train()
            
            temp_image_head = copy.deepcopy(self.image_perception_head).to(self.device)
            spatial_processor = SpatialProcessor(embed_dim=temp_model.output_feature_dim).to(self.device)
            temp_regression_head = copy.deepcopy(self.regression_head).to(self.device)
            
            # --- 3. Define optimizer ---
            trainable_params = list(temp_image_head.parameters()) + \
                               list(spatial_processor.parameters()) + \
                               list(temp_model.parameters()) + \
                               list(temp_regression_head.parameters())
            
            optimizer = optim.AdamW(trainable_params, lr=self.benchmark_config['learning_rate'] * 0.1)
            loss_fn = nn.MSELoss()
            
            # --- 4. Training Loop ---
            temp_image_head.train()
            spatial_processor.train()
            temp_regression_head.train()
            num_training_steps = 2000
            
            dataset_train = TensorDataset(videos, true_coords)
            loader_train = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'] // 2, shuffle=True)
            
            step = 0
            while step < num_training_steps:
                try:
                    for batch_videos, batch_coords in loader_train:
                        if step >= num_training_steps: break
                        
                        batch_size_B_eff = batch_videos.shape[0]
                        optimizer.zero_grad()
                        
                        video_frames_batch = batch_videos.view(batch_size_B_eff * T, C, H, W)
                        patch_sequences = temp_image_head(video_frames_batch)
                        frame_features = spatial_processor(patch_sequences)
                        temporal_sequence = frame_features.view(batch_size_B_eff, T, -1)
                        model_output_features = temp_model(x=None, embeds=temporal_sequence, return_features=True)
                        video_feature = model_output_features.mean(dim=1)
                        
                        # <<< FINAL FIX: Apply sigmoid BEFORE calculating loss
                        predicted_coords = torch.sigmoid(temp_regression_head(video_feature))
                        
                        loss = loss_fn(predicted_coords, batch_coords)
                        
                        if not torch.isnan(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                            optimizer.step()
                        
                        step += 1
                except StopIteration:
                    continue

            # --- 5. Evaluation Loop ---
            temp_model.eval()
            temp_image_head.eval()
            spatial_processor.eval()
            temp_regression_head.eval()
            eval_losses = []
            with torch.no_grad():
                loader_eval = DataLoader(dataset_train, batch_size=self.benchmark_config['batch_size'] // 2)
                for batch_eval_videos, batch_eval_coords in loader_eval:
                    E_B_eff = batch_eval_videos.shape[0]
                    eval_frames = batch_eval_videos.view(E_B_eff * T, C, H, W)
                    eval_patch_sequences = temp_image_head(eval_frames)
                    eval_frame_features = spatial_processor(eval_patch_sequences)
                    eval_temporal_sequence = eval_frame_features.view(E_B_eff, T, -1)
                    eval_model_features = temp_model(x=None, embeds=eval_temporal_sequence, return_features=True)
                    eval_video_feature = eval_model_features.mean(dim=1)
                    
                    # <<< FINAL FIX: Sigmoid is already applied here, so this is consistent now.
                    eval_predicted_coords = torch.sigmoid(temp_regression_head(eval_video_feature))
                    
                    distance_eval = torch.abs(eval_predicted_coords - batch_eval_coords).sum(dim=1)
                    eval_losses.extend(distance_eval.cpu().numpy())

            # --- 6. Calculate score ---
            score = math.exp(-2.0 * np.mean(eval_losses)) if eval_losses else 0.0
            level_scores[level] = score
            print(f"    - Level {level} ({name}) Score: {score:.4f}")

        # --- 7. Calculate final score ---
        weighted_sum = sum(weights.get(l, 0) * s for l, s in level_scores.items())
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Final Trajectory_Prediction_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _run_probe_training(self, model, probe_data):
        dataset = TensorDataset(probe_data['sequences'], probe_data['memory_labels'], probe_data['reasoning_labels'])
        loader = DataLoader(dataset, batch_size=self.benchmark_config['batch_size'], shuffle=True)
        
        # --- PROBE HEAD OPTIMIZATION ---
        # Include perception heads in the optimizer to allow them to adapt to the model
        all_params = list(model.parameters()) + list(self.image_perception_head.parameters()) + list(self.audio_perception_head.parameters()) + list(self.regression_head.parameters()) + list(self.multidomain_classification_head.parameters())
        optimizer = optim.AdamW(all_params, lr=self.benchmark_config['learning_rate'])
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        model.train()
        step, training_complete, probe_results = 0, False, {'train_loss': [], 'val_loss': 0.0, 'val_accuracy': 0.0, 'learning_headroom': 0.0}
        while not training_complete:
            for seqs, mem_labels, reas_labels in loader:
                seqs, mem_labels, reas_labels = seqs.to(self.device), mem_labels.to(self.device), reas_labels.to(self.device)
                optimizer.zero_grad()
                logits = model(seqs)
                loss_mem = loss_fn(logits.view(-1, self.benchmark_config['num_classes']), mem_labels.view(-1))
                loss_reas = loss_fn(logits.view(-1, self.benchmark_config['num_classes']), reas_labels.view(-1))
                total_loss = loss_mem + loss_reas
                if not torch.isnan(total_loss):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                probe_results['train_loss'].append(total_loss.item())
                step += 1
                if step >= self.benchmark_config['probe_training_steps']:
                    training_complete = True
                    break
        probe_results['learning_headroom'] = self._compute_learning_headroom(probe_results['train_loss'])
        model.eval()
        with torch.no_grad():
            correct_mem, total_mem, val_loss_sum, batches = 0, 0, 0, 0
            for seqs, mem_labels, reas_labels in loader:
                seqs, mem_labels, reas_labels = seqs.to(self.device), mem_labels.to(self.device), reas_labels.to(self.device)
                logits = model(seqs)
                preds, mask = torch.argmax(logits, dim=-1), mem_labels != -100
                correct_mem += (preds[mask] == mem_labels[mask]).sum().item()
                total_mem += mask.sum().item()
                loss_mem = loss_fn(logits.view(-1, self.benchmark_config['num_classes']), mem_labels.view(-1))
                loss_reas = loss_fn(logits.view(-1, self.benchmark_config['num_classes']), reas_labels.view(-1))
                if not torch.isnan(loss_mem) and not torch.isnan(loss_reas):
                    val_loss_sum += (loss_mem + loss_reas).item()
                    batches += 1
            probe_results['val_accuracy'] = correct_mem / total_mem if total_mem > 0 else 0.0
            probe_results['val_loss'] = val_loss_sum / batches if batches > 0 else float('inf')
        return probe_results

    def _compute_parameter_score(self):
        # --- MODEL CONFIGURATION INJECTION ---
        full_model_config = {**self.model_config, **self.benchmark_config}
        model = self.model_class(full_model_config)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Avoid division by zero
        if num_params == 0:
            return 0.0
        return 1.0 / (1.0 + (num_params / 1_000_000.0))

    def _compute_scalability_score(self):
        times, seq_lengths = [], np.array([64, 128, 256])
        # --- MODEL CONFIGURATION INJECTION ---
        full_model_config = {**self.model_config, **self.benchmark_config}
        model_instance = self.model_class(full_model_config).to(self.device).eval()
        with torch.no_grad():
            for sl in seq_lengths:
                try:
                    # Use a smaller batch size for longer sequences to avoid OOM
                    effective_batch_size = max(1, self.benchmark_config['batch_size'] // (sl // 64))
                    inp = torch.randint(0, self.benchmark_config['vocab_size'], (effective_batch_size, int(sl)), device=self.device)
                    for _ in range(5): model_instance(inp) # Warmup
                    start_time = time.time()
                    for _ in range(10): model_instance(inp)
                    times.append((time.time() - start_time) / 10)
                except Exception as e:
                    print(f"Scalability test failed for length {sl}: {e}")
                    times.append(1e9) # Assign a large time for failure
        if any(t > 1e8 for t in times):
            return 0.0
        log_lengths, log_times = np.log(seq_lengths), np.log(np.array(times) + 1e-9)
        try:
            # Use lstsq for numerical stability
            A = np.vstack([log_lengths, np.ones(len(log_lengths))]).T
            beta, _ = np.linalg.lstsq(A, log_times, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = 5.0 # Default to a high complexity if fitting fails
        t_median = times[1] if len(times) > 1 else times[0] if times else 1e9
        s_complexity = math.exp(-0.7 * max(0, beta - 1.0))
        s_speed = math.exp(-t_median / 0.05) # Adjusted scaling factor
        return s_complexity * s_speed

    def _compute_interpretability_score(self, model, probe_data):
        interpret_data = generate_composite_probe_data(self.benchmark_config['batch_size'], self.benchmark_config['sequence_length'], self.benchmark_config['vocab_size'], self.benchmark_config['num_classes'], mem_difficulty_levels=[1], reas_difficulty_levels=[])
        loader = DataLoader(TensorDataset(*interpret_data.values()), batch_size=self.benchmark_config['batch_size'])
        try:
            seqs, labels, _, _, _ = next(iter(loader))
        except StopIteration:
            return 0.0 # No data to process
        seqs, labels = seqs.to(self.device), labels.to(self.device)
        with torch.no_grad():
            logits = model(seqs)
            probs = F.softmax(logits, dim=-1)
            recall_mask = labels != -100
            if not recall_mask.any():
                 return 0.0
            correct_class_indices = labels[recall_mask]
            p_originals = probs[recall_mask].gather(1, correct_class_indices.unsqueeze(1)).squeeze()
        recall_token_id, copy_token_min_id, copy_token_max_id = 23, 10, 10 + self.benchmark_config['num_classes']
        signal_indices = [torch.where((seqs[i] >= copy_token_min_id) & (seqs[i] < copy_token_max_id))[0] for i in range(seqs.size(0))]
        recall_indices = [torch.where(seqs[i] == recall_token_id)[0] for i in range(seqs.size(0))]
        noise_impacts, signal_impacts = [], []
        num_noise_tokens_to_test = 5
        for i in range(seqs.size(0)):
            if not hasattr(p_originals, "__len__") or i >= len(p_originals) or len(recall_indices[i]) == 0:
                continue
            p_original = p_originals[i].item()
            if p_original < 0.5:
                continue
            signal_idxs_to_test = torch.cat([signal_indices[i], recall_indices[i]])
            if len(signal_idxs_to_test) == 0:
                continue
            current_signal_impacts = []
            for idx in signal_idxs_to_test:
                seq_perturbed = seqs[i].clone()
                seq_perturbed[idx] = 0
                with torch.no_grad():
                    p_new = F.softmax(model(seq_perturbed.unsqueeze(0)), dim=-1)[0, recall_indices[i][0], correct_class_indices[i]]
                    current_signal_impacts.append(p_original - p_new.item())
            if current_signal_impacts:
                signal_impacts.append(np.mean(current_signal_impacts))
            current_noise_impacts = []
            all_indices = set(range(seqs.size(1)))
            signal_idx_set = set(signal_idxs_to_test.tolist())
            try:
                population = list(all_indices - signal_idx_set)
                if len(population) < num_noise_tokens_to_test:
                    continue
                noise_indices = random.sample(population, num_noise_tokens_to_test)
            except ValueError: # Not enough items to sample
                continue
            for idx in noise_indices:
                seq_perturbed = seqs[i].clone()
                seq_perturbed[idx] = 0
                with torch.no_grad():
                    p_new = F.softmax(model(seq_perturbed.unsqueeze(0)), dim=-1)[0, recall_indices[i][0], correct_class_indices[i]]
                    current_noise_impacts.append(abs(p_original - p_new.item()))
            if current_noise_impacts:
                noise_impacts.append(np.mean(current_noise_impacts))
        if not signal_impacts or not noise_impacts:
            return 0.0
        mean_signal_impact = np.mean(signal_impacts)
        mean_noise_impact = np.mean(noise_impacts)
        # Avoid division by zero
        focus_ratio = max(0, mean_signal_impact) / (mean_noise_impact + 1e-9)
        return 1.0 / (1.0 + math.exp(-0.5 * (math.log(focus_ratio + 1) - math.log(10))))

    def _evaluate_multi_level(self, model, probe_data, task_type):
        if task_type == 'memory':
            difficulty_tensor, labels_tensor, weights = probe_data['memory_difficulty'], probe_data['memory_labels'], {1: 10, 2: 8, 3: 6, 5: 4, 8: 2}
        else:
            difficulty_tensor, labels_tensor, weights = probe_data['reasoning_difficulty'], probe_data['reasoning_labels'], {1: 2, 2: 4, 3: 6}
        difficulty_levels = sorted([l for l in difficulty_tensor.unique().tolist() if l != 0]) # Filter out 0 difficulty
        correct_counts = {level: 0 for level in difficulty_levels}
        total_counts = {level: 0 for level in difficulty_levels}
        dataset = TensorDataset(probe_data['sequences'], labels_tensor, difficulty_tensor)
        loader = DataLoader(dataset, batch_size=self.benchmark_config['batch_size'])
        with torch.no_grad():
            for seqs, labels, difficulties in loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                preds = torch.argmax(model(seqs), dim=-1)
                mask = labels != -100
                for level in difficulty_levels:
                    # Ensure level is a tensor for comparison
                    level_tensor = torch.tensor(level, device=difficulties.device)
                    level_mask = (difficulties == level_tensor).unsqueeze(1) & mask.cpu()
                    correct_counts[level] += (preds.cpu()[level_mask] == labels.cpu()[level_mask]).sum().item()
                    total_counts[level] += level_mask.sum().item()
        # Avoid division by zero
        accuracies = {level: correct_counts[level] / total_counts[level] if total_counts[level] > 0 else 0 for level in difficulty_levels}
        return accuracies, weights

    def _compute_memory_score(self, model, probe_data):
        accuracies, weights = self._evaluate_multi_level(model, probe_data, 'memory')
        print(f"  > Memory Accuracy per Repetition Level: { {k: round(v, 4) for k, v in accuracies.items()} }")
        weighted_sum = sum(weights.get(level, 0) * accuracies.get(level, 0) for level in weights)
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Memory_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _compute_reasoning_score(self, model, probe_data):
        accuracies, weights = self._evaluate_multi_level(model, probe_data, 'reasoning')
        print(f"  > Reasoning Accuracy per Difficulty Level: { {k: round(v, 4) for k, v in accuracies.items()} }")
        normalized_accuracies = {}
        choices = {1: 2, 2: 3, 3: 4}
        for level, acc in accuracies.items():
            baseline = 1.0 / choices.get(level, 2)
            # Avoid division by zero
            denom = (1.0 - baseline)
            if denom > 0:
                normalized_accuracies[level] = max(0.0, (acc - baseline) / denom)
            else:
                normalized_accuracies[level] = 0.0
        weighted_sum = sum(weights.get(level, 0) * normalized_accuracies.get(level, 0) for level in weights)
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        print(f"  > Reasoning_Score (Weighted) calculated: {final_score:.4f}")
        return final_score

    def _compute_robustness_score(self, model, probe_data, pr):
        noise_levels, weights, level_scores = [0.05, 0.15, 0.30], {0.05: 1, 0.15: 2, 0.30: 3}, {}
        print(f"  > Robustness Score per Noise Level:")
        for level in noise_levels:
            noisy_sequences = probe_data['sequences'].clone()
            mask = (noisy_sequences >= 30)
            noise = torch.randint(30, self.benchmark_config['vocab_size'], noisy_sequences.shape)
            perturb_mask = (torch.rand(noisy_sequences.shape) < level) & mask
            noisy_sequences[perturb_mask] = noise[perturb_mask]
            noisy_dataset = TensorDataset(noisy_sequences, probe_data['memory_labels'], probe_data['reasoning_labels'])
            noisy_loader = DataLoader(noisy_dataset, batch_size=self.benchmark_config['batch_size'])
            loss_fn, noisy_loss_sum, batches = nn.CrossEntropyLoss(ignore_index=-100), 0, 0
            with torch.no_grad():
                for seqs, mem_labels, reas_labels in noisy_loader:
                    seqs, mem_labels, reas_labels = seqs.to(self.device), mem_labels.to(self.device), reas_labels.to(self.device)
                    logits = model(seqs)
                    loss_m = loss_fn(logits.view(-1, self.benchmark_config['num_classes']), mem_labels.view(-1))
                    loss_r = loss_fn(logits.view(-1, self.benchmark_config['num_classes']), reas_labels.view(-1))
                    if not torch.isnan(loss_m) and not torch.isnan(loss_r):
                        noisy_loss_sum += (loss_m + loss_r).item()
                        batches += 1
            noisy_loss = noisy_loss_sum / batches if batches > 0 else float('inf')
            scaler = max(0.0, (pr['val_accuracy'] - 0.5) * 2)
            # Avoid division by zero
            penalty_val_denom = (noisy_loss + pr['val_loss'] + 1e-9)
            if penalty_val_denom > 0:
                penalty_val = max(0, noisy_loss - pr['val_loss']) / penalty_val_denom
            else:
                penalty_val = 0
            penalty = math.sqrt(penalty_val) if penalty_val > 0 else 0
            level_scores[level] = scaler * (1.0 - penalty)
            print(f"    - Noise {int(level * 100)}%: {level_scores[level]:.4f}")
        weighted_sum = sum(weights.get(level, 0) * level_scores.get(level, 0) for level in weights)
        total_weight = sum(weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return final_score

    def _compute_generalization_score(self, pr):
        if not pr['train_loss'] or pr['val_loss'] == float('inf'):
             return 0.0
        train_loss_mean = np.mean(pr['train_loss'])
        # Avoid division by zero
        penalty_val_denom = (pr['val_loss'] + train_loss_mean + 1e-9)
        if penalty_val_denom > 0:
            penalty_val = max(0, pr['val_loss'] - train_loss_mean) / penalty_val_denom
        else:
            penalty_val = 0
        penalty = math.sqrt(penalty_val) if penalty_val > 0 else 0
        return pr['val_accuracy'] * (1.0 - penalty)

    def _compute_uncertainty_score(self):
        confusing_data = generate_composite_probe_data(self.benchmark_config['batch_size'], self.benchmark_config['sequence_length'], self.benchmark_config['vocab_size'], self.benchmark_config['num_classes'], reas_difficulty_levels=[])
        confusing_input = confusing_data['sequences'].to(self.device)
        confusing_input[confusing_input == 23] = 0 # Remove recall token to make task confusing
        # --- MODEL CONFIGURATION INJECTION ---
        full_model_config = {**self.model_config, **self.benchmark_config}
        model = self.model_class(full_model_config).to(self.device).eval()
        with torch.no_grad():
            logits = model(confusing_input)
            probs = torch.softmax(logits, dim=-1)
            # Calculate entropy for each sample, then average
            avg_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        max_entropy = np.log(self.benchmark_config['num_classes'])
        # Avoid division by zero
        if max_entropy > 0:
            return avg_entropy / max_entropy
        else:
            return 0.0

    def _compute_continual_learning_score(self):
        # --- MODEL CONFIGURATION INJECTION ---
        full_model_config = {**self.model_config, **self.benchmark_config}
        temp_model = self.model_class(full_model_config).to(self.device)
        optimizer = optim.AdamW(temp_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        data_A = generate_composite_probe_data(256, self.benchmark_config['sequence_length'], self.benchmark_config['vocab_size'], self.benchmark_config['num_classes'], mem_difficulty_levels=[3], reas_difficulty_levels=[])
        loader_A = DataLoader(TensorDataset(data_A['sequences'], data_A['memory_labels']), batch_size=32)
        data_B = generate_composite_probe_data(256, self.benchmark_config['sequence_length'], self.benchmark_config['vocab_size'], self.benchmark_config['num_classes'], mem_difficulty_levels=[], reas_difficulty_levels=[1])
        loader_B = DataLoader(TensorDataset(data_B['sequences'], data_B['reasoning_labels']), batch_size=32)
        temp_model.train()
        for i, (seqs, labels) in enumerate(loader_A):
            if i > 20: break
            seqs, labels = seqs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = loss_fn(temp_model(seqs).view(-1, self.benchmark_config['num_classes']), labels.view(-1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc_A1 = self._evaluate_on_loader(temp_model, loader_A, 'memory')
        temp_model.train()
        for i, (seqs, labels) in enumerate(loader_B):
            if i > 20: break
            seqs, labels = seqs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            loss = loss_fn(temp_model(seqs).view(-1, self.benchmark_config['num_classes']), labels.view(-1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc_A2 = self._evaluate_on_loader(temp_model, loader_A, 'memory')
        # Avoid division by zero
        forgetting_factor_denom = (acc_A1 + 1e-9)
        if forgetting_factor_denom > 0:
            forgetting_factor = max(0, acc_A1 - acc_A2) / forgetting_factor_denom
        else:
            forgetting_factor = 0
        return 1.0 - forgetting_factor

    def _compute_learning_headroom(self, all_train_losses):
        if len(all_train_losses) < 50:
            return 0.0
        last_losses = all_train_losses[-50:]
        steps = np.arange(len(last_losses))
        try:
            slope, _ = np.polyfit(steps, last_losses, 1)
        except np.linalg.LinAlgError: # Handle case where fitting fails
            return 0.0
        return 1.0 / (1.0 + math.exp(slope * 100))

    def _evaluate_on_loader(self, model, loader_or_data, task_type):
        if isinstance(loader_or_data, dict):
            labels_tensor = loader_or_data['memory_labels'] if task_type == 'memory' else loader_or_data['reasoning_labels']
            dataset = TensorDataset(loader_or_data['sequences'], labels_tensor)
            loader = DataLoader(dataset, batch_size=self.benchmark_config['batch_size'])
        else:
            loader = loader_or_data
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                seqs, labels = batch[0], batch[1]
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                preds = torch.argmax(model(seqs), dim=-1)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        # Avoid division by zero
        return correct / total if total > 0 else 0.0

# --- Optional CV2 Handling ---
try:
    import cv2
except ImportError:
    print("Warning: OpenCV is not installed. Image/Video probes will be less effective.")
    print("Please install it using: pip install opencv-python")

    class DummyCV2:
        def line(self, *args, **kwargs):
            pass

        def resize(self, arr, size):
            # Simple crop if size is smaller, pad if larger (basic simulation)
            h, w = arr.shape[:2]
            new_h, new_w = size
            if new_h <= h and new_w <= w:
                return arr[:new_h, :new_w]
            else:
                # Pad with zeros if needed (basic simulation)
                pad_h = max(0, new_h - h)
                pad_w = max(0, new_w - w)
                return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

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
        'Texture_Detection_Score': 'TXT',
        'Energy_Envelope_Score': 'ENGY',
        'Prosody_Detection_Score': 'PRDY',
        'Formant_Dynamics_Score': 'FRMN',
        'Trajectory_Prediction_Score': 'TRAJ'
    }
    # Ensure all metrics exist in the dataframe, fill missing with 0
    for metric in metrics_to_plot.keys():
        if metric not in df.index:
            df.loc[metric] = 0.0

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
    print(f"Starting IQ-NET Profiler on device: {BENCHMARK_CONFIG['device']}")
    
    # --- MODEL SELECTION ---
    # Define the models to test by mapping their name to their class
    models_to_test = {
        "Zarvan": ZarvanModel,
        "LSTM": LSTMModel,
        "GRU": GRUModel,
        "Transformer": SimpleTransformerModel,
        "CNN_1D": CNN1DModel,
        # Add other models here, e.g.,
        # "SimpleRNN": SimpleRNNModel,
    }
    
    results = {}
    for name, model_class in models_to_test.items():
        # Get the specific config for this model
        if name not in MODEL_CONFIGS:
            print(f"Warning: No configuration found for model '{name}'. Skipping.")
            continue
            
        model_config = MODEL_CONFIGS[name]
        
        # Create and run the profiler
        profiler = HolisticAptitudeProfiler(model_class, name, BENCHMARK_CONFIG, model_config)
        profile_scores = profiler.get_profile()
        results[name] = profile_scores

    print("\n" + "=" * 20 + " FINAL IQ-NET EXPANDED REPORT " + "=" * 20)
    if not results:
        print("No models were profiled successfully.")
        exit()

    report_df = pd.DataFrame(results).fillna(0)
    report_df.index.name = 'Metric'
    
    # --- WEIGHTS FOR FINAL IQ SCORE ---
    weights = {
        'Memory_Score': 20, 
        'Reasoning_Score': 20, 
        'Scalability_Score': 10,
        'Robustness_Score': 10, 
        'Generalization_Score': 5, 
        'Learning_Headroom': 5,
        'Parameter_Score': 5, 
        'Continual_Learning_Score': 5, 
        'Interpretability_Score': 5,
        'Uncertainty_Score': 5, 
        'Spatial_Focus_Score': 10, 
        'Pattern_Invariance_Score': 10,
        'Frequency_Detection_Score': 10, 
        'Rhythm_Comprehension_Score': 10,
        'Texture_Detection_Score': 5,
        'Energy_Envelope_Score': 5,
        'Prosody_Detection_Score': 5,
        'Formant_Dynamics_Score': 5,
        'Trajectory_Prediction_Score': 10
    }
    
    # Ensure all weighted metrics are present in the dataframe
    for col in weights.keys():
        if col not in report_df.index:
            report_df.loc[col] = 0.0
            
    final_scores = {}
    total_weight = sum(weights.values())
    
    # Calculate final IQ score for each model
    for model_name in report_df.columns:
        model_scores = report_df[model_name]
        weighted_sum = sum(model_scores.get(metric, 0) * w for metric, w in weights.items())
        final_scores[model_name] = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0

    report_df.loc['Final_IQ_Score'] = final_scores
    
    # Sort models by final IQ score (descending)
    report_df = report_df.reindex(columns=sorted(report_df.columns, key=lambda c: report_df.loc['Final_IQ_Score', c], reverse=True))
    
    # Print the final report
    print(report_df.round(4).to_markdown())
    
    # Generate and save radar charts
    plot_radar_charts(report_df)
