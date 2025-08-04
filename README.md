# IQ-NET for Sequences

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official implementation for **IQ-NET for Sequences**, a holistic and interpretable framework for evaluating sequence modeling architectures, as presented in our paper.

The standard evaluation of deep learning models often relies on a narrow set of metrics like accuracy. IQ-NET provides a multi-faceted assessment, offering deeper insights into a model's true capabilities, including its efficiency, robustness, and the quality of its internal representations.

---

## Citation

If you use this framework or code in your research, please cite our paper:

@article{sajjadi2025zarvan,
title={Zarvan: An Efficient Gated Architecture for Sequence Modeling with Linear Complexity},
author={Sajjadi, Yasser},
journal={Preprints.org},
year={2025},
doi={10.20944/preprints202507.2512.v1}
}

---

## Features

IQ-NET for Sequences evaluates models across nine distinct, scientifically-grounded metrics:

* 🧠 **Processing IQ (Proc. IQ):** Measures the quality of the model's internal transformations.
* 🛡️ **Adversarial Robustness Index (ARI):** Assesses resilience to noisy inputs.
* ⚖️ **Stability:** Quantifies performance consistency across varying data difficulty.
* 🚀 **Learning Speed Index (LSI):** Measures how efficiently the model learns.
* ⚙️ **Efficiency Complexity Index (ECI):** Evaluates accuracy per million parameters.
* 📈 **Efficiency Scalability Index (ESI):** Assesses how inference time scales with sequence length.
* 🌐 **Generalization Transfer Index (GTI):** Measures the gap between training and test performance.
* ✨ **Output Novelty Ratio (ONR):** Evaluates the diversity and confidence of model outputs.
* 🤔 **Reasoning Index (REI):** Probes the model's capacity for iterative refinement.

---

## Visualizing the Results

The framework generates radar charts to provide an intuitive, at-a-glance understanding of each model's unique performance profile.

![Combined IQ Profile](text_iq_profile_combined.png)
*Figure: A combined view of the IQ profiles for all evaluated models, highlighting their relative strengths and weaknesses.*

---

## Repository Structure

This repository is organized as follows:

* `benchmark_text.py`: The main script to run the full IQ-NET evaluation on seven different text models (CNN, TCN, LSTM, GRU, Transformer, Longformer, Zarvan).
* `standard_benchmark.py`: A script to run a standard, accuracy-focused benchmark on the same set of models.
* `benchmark_text_result.txt`: A sample output log file from running `benchmark_text.py`.
* `standard_benchmark_result.txt`: A sample output log file from running `standard_benchmark.py`.
* `*.png`: Radar chart visualizations of the evaluation results.

---

## Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* A CUDA-enabled GPU is highly recommended for reasonable training times.

### Installation

1.  **Clone the repository:**
    ```
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required dependencies:**
    It is recommended to use a virtual environment.
    ```
    pip install -r requirements.txt
    ```
    If a `requirements.txt` file is not available, you can install the packages manually:
    ```
    pip install torch transformers datasets pandas scikit-learn matplotlib scipy
    ```

### Usage

You can run the evaluations using the provided scripts. The scripts will download the IMDB dataset automatically on the first run.

1.  **Run the full IQ-NET Benchmark:**
    This will evaluate all models, save a detailed report (`text_iq_report.csv`), and generate the radar charts (`.png` files).
    ```
    python benchmark_text.py
    ```

2.  **Run the Standard Benchmark:**
    This will evaluate all models based on standard metrics (Accuracy, F1, etc.), and save a report (`standard_benchmark_report.csv`).
    ```
    python standard_benchmark.py
    ```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this code for your own research and applications.
