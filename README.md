# IQ-NET: Holistic Aptitude Profiler for Neural Network Architectures

**IQ-NET** is a lightweight, rapid, and comprehensive framework for profiling the intrinsic capabilities of neural network architectures. Unlike traditional benchmarks like GLUE or ImageNet, which focus on single-task performance and require extensive GPU resources, IQ-NET evaluates models across 15 theoretically grounded metrics in minutes using CPU-based synthetic probe tasks. It spans text, image, audio, and video domains, revealing each model's unique "personality profile" through vivid radar charts. This project empowers researchers to select task-optimal models, identify improvement areas, and drive architectural innovation.

The code evaluates six neural network models—LSTM, GRU, Transformer, TCN, CNN, and Zarvan—across metrics like Reasoning (RSN), Memory (MEM), Scalability (SCL), Robustness (ROB), Generalization (GEN), and more. The results are visualized as individual and combined radar charts, saved as high-resolution PNG files.

## Features
- **Rapid Profiling**: Evaluates models in minutes using CPU-based synthetic tasks, ensuring accessibility and fairness.
- **Comprehensive Metrics**: Profiles 15 aptitudes, including:
  - **RSN**: Reasoning (stateful logical processing)
  - **MEM**: Memory (information recall under noise)
  - **SCL**: Scalability (handling increasing sequence lengths)
  - **ROB**: Robustness (performance under noise)
  - **GEN**: Generalization (performance on unseen data)
  - **HEAD**: Learning Headroom (remaining learning potential)
  - **PAR**: Parameter Efficiency (fewer parameters)
  - **INTP**: Interpretability (focus on relevant inputs)
  - **UNC**: Uncertainty (appropriate uncertainty on ambiguous inputs)
  - **CONT**: Continual Learning (knowledge retention across tasks)
  - **SPF**: Spatial Focus (feature localization in images)
  - **PAT**: Pattern Invariance (recognition despite transformations)
  - **FREQ**: Frequency Detection (identifying frequencies in audio)
  - **RHY**: Rhythm Comprehension (classifying temporal patterns)
  - **TRAJ**: Trajectory Prediction (extrapolating object motion in videos)
- **Visualization**: Generates radar charts for each model and a combined comparison, saved as `iq_profile_<model>.png` and `iq_net_radar_comparison.png`.
- **Reproducible**: Includes seed setting for consistent results and synthetic datasets for bias-free evaluation.
- **Open Source**: All code and synthetic probe datasets follow FAIR data principles.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Dependencies
Install the required Python libraries using:
```bash
pip install torch numpy scipy pandas matplotlib opencv-python
```
**Note**: `opencv-python` is optional but required for image and video probe tasks. If not installed, the script gracefully handles the absence with a fallback mechanism.

### Clone the Repository
```bash
git clone https://github.com/systbs/iq-net.git
cd iq-net
```

## Usage
1. **Run the Profiler**:
   Execute the main script to profile the six models (LSTM, GRU, Transformer, TCN, CNN, Zarvan):
   ```bash
   python benchmark.py
   ```
   The script will:
   - Profile each model across the 15 metrics.
   - Output detailed logs to the console, including per-metric scores and sub-level accuracies.
   - Generate radar charts saved as PNG files in the project directory.

2. **Output Files**:
   - **Individual Radar Charts**: `iq_profile_<model>.png` (e.g., `iq_profile_lstm.png`) for each model's profile.
   - **Combined Radar Chart**: `iq_net_radar_comparison.png` comparing all models.
   - **Console Output**: A markdown-formatted table summarizing the results, including a weighted Final IQ Score (scaled to 100).

3. **Example Output**:
   ```
   Starting IQ-NET Profiler on device: cpu
   --- Profiling LSTM ---
   Running static tests...
     > Parameter_Score calculated: 0.8070
     > Scalability_Score calculated: 0.6830
   ...
   ==================== FINAL IQ-NET EXPANDED REPORT ====================
   | Metric                     |   LSTM |   GRU | Transformer |   TCN |   CNN | Zarvan |
   |----------------------------|--------|-------|-------------|-------|-------|--------|
   | Memory_Score               | 0.9842 | ...   | ...         | ...   | ...   | ...    |
   ...
   | Final_IQ_Score             | ...    | ...   | ...         | ...   | ...   | ...    |
   ```

## Project Structure
- `benchmark.py`: Main script implementing the IQ-NET framework, including model definitions, probe tasks, and visualization.
- `iq_profile_<model>.png`: Output radar charts for each model.
- `iq_net_radar_comparison.png`: Combined radar chart comparing all models.

## Methodology
IQ-NET uses synthetic probe tasks to evaluate models across four domains:
- **Text**: Memory and reasoning tasks with synthetic sequences (vocab_size=100, sequence_length=128, num_samples=1200).
- **Image**: Spatial focus and pattern invariance tasks using 32x32 images.
- **Audio**: Frequency detection and rhythm comprehension tasks with 2048-length signals.
- **Video**: Trajectory prediction tasks using 16-frame 32x32 videos.

Standardized perception heads (Micro-CNN for images, Micro-1D-CNN for audio) ensure fair evaluation of reasoning capabilities. The framework runs on a CPU for accessibility and normalizes resources to eliminate hardware bias. Metrics are weighted to reflect real-world priorities (e.g., Memory: 20, Scalability: 10).

## Results
IQ-NET reveals distinct architectural personalities:
- **Zarvan & Transformer**: Excel in Memory (1.0000), Generalization (1.0000), and Robustness (~0.9), ideal for multi-domain tasks.
- **LSTM & GRU**: Dominate Reasoning (0.6141, 0.8120) and Rhythm Comprehension, suited for sequential tasks like dialogue systems.
- **CNN**: Leads in Scalability (0.9221) and Continual Learning (0.9268), perfect for edge devices.
- **TCN**: Strong in Memory (0.9969) but struggles with audio tasks, highlighting areas for improvement.

See the generated radar charts for a visual comparison of model profiles.

## Contributing
Contributions are welcome! Please submit issues or pull requests for bug fixes, feature additions, or documentation improvements. Follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, contact Yasser Sajjadi at [yassersajjadi@gmail.com](mailto:yassersajjadi@gmail.com).

---

*Built with ❤️ by Yasser Sajjadi, powered by synthetic probe tasks and a passion for unveiling neural network potential.*
