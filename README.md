<div align="center">

# 🧠 RGF: Residual Graph Fusion

### *A Lightweight Gated Architecture for Node Classification on Noisy Heterophilic Graphs*

<br>

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/vaniakarimi)

</div>

---

## 📌 Overview

**Residual Graph Fusion (RGF)** is a lightweight Graph Neural Network designed to simultaneously address **two critical challenges** in node classification:

| Challenge | Solution |
|:----------|:---------|
| 🔗 **Heterophilic Graphs** (connected nodes have different labels) | Learnable gate adaptively balances local vs neighborhood information ($g \approx 0.73$) |
| 🛡️ **Feature Noise** (corrupted node attributes) | Residual connection preserves clean signals ($+36.67\%$ robustness) |

---

## 🎯 Key Results

<div align="center">

| Dataset | Type | RGF Accuracy | Improvement over GCN |
|:--------|:----:|-------------:|---------------------:|
| **Wisconsin** | Heterophilic | **71.76% ± 2.80%** | **+32.35%** |
| **Texas** | Heterophilic | **72.97% ± 2.96%** | **+41.44%** |
| **Cornell** | Heterophilic | **61.08% ± 2.16%** | **+25.94%** |
| **Chameleon** | Heterophilic | **51.49% ± 0.90%** | **+19.99%** |
| **Squirrel** | Heterophilic | **33.45% ± 0.64%** | **+6.15%** |

</div>

**📊 Under 30% Gaussian feature noise:** RGF maintains **66.67%** accuracy while GCN drops to **30.0%** → **+36.67% absolute improvement**

---

## 🏗️ Architecture

<div align="center">
<img src="figures/rgf_architecture_final.png" width="700">
</div>

### Mathematical Formulation

| Step | Equation |
|:----:|:---------|
| **Feature Transformation** | $\mathbf{H}_{\text{local}} = \text{ReLU}(\mathbf{X}\mathbf{W}_1),\quad \mathbf{H}_{\text{neigh}} = \text{ReLU}(\hat{\mathbf{A}}\mathbf{X}\mathbf{W}_1)$ |
| **Gated Fusion** | $\mathbf{g} = \sigma([\mathbf{H}_{\text{local}} \| \mathbf{H}_{\text{neigh}}]\mathbf{W}_2),\quad \mathbf{H} = \mathbf{g} \odot \mathbf{H}_{\text{local}} + (\mathbf{1}_n - \mathbf{g}) \odot \mathbf{H}_{\text{neigh}}$ |
| **Classification** | $\hat{\mathbf{Y}} = \text{softmax}(\mathbf{H}\mathbf{W}_3)$ |

---

## 📈 Visual Results

<div align="center">
<table>
<tr>
<td align="center"><img src="figures/noise_sensitivity.png" width="250"><br><b>Noise Robustness</b></td>
<td align="center"><img src="figures/accuracy_vs_params.png" width="250"><br><b>Accuracy vs Parameters</b></td>
<td align="center"><img src="figures/gate_distribution.png" width="250"><br><b>Gate Distribution (μ=0.729)</b></td>
</tr>
</table>
</div>

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/vaniakarimi/RGF-Noise-Resilient.git
cd RGF-Noise-Resilient
pip install -r requirements.txt
