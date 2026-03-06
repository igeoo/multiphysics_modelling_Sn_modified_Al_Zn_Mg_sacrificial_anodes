# Multi-Physics Modelling of Sn-Modified Al-Zn-Mg Sacrificial Anodes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the computational framework for a predictive multi-physics model of Al-Zn-Mg-Sn sacrificial anodes. The framework integrates metallurgical precipitation kinetics with electrochemical dissolution dynamics to provide accurate service-life predictions for cathodic protection systems.

The model is designed to bridge the gap between microstructure evolution (aging) and electrochemical performance, specifically focusing on the role of Trace Element (Sn) modifications.

## Key Physical Components

### 1. Precipitation Kinetics (JMAK)
The model utilizes a **Johnson-Mehl-Avrami-Kolmogorov (JMAK)** framework to describe the transformed volume fraction ($X$) of precipitates during thermal aging:
$$X(t, T) = 1 - \exp(-(k(T) \cdot t)^n)$$
where $k(T)$ follows an Arrhenius dependency.

### 2. Electrochemical Dissolution (Butler-Volmer)
The electrochemical behavior is governed by **Butler-Volmer kinetics**, modified with a coupling term for precipitate volume fraction and Sn composition:
$$i_0 = i_{0,base} \cdot (1 + \beta X^m) \cdot (1 + \gamma C_{Sn})$$

### 3. Service Life Integration
Long-term mass loss and geometry evolution are calculated via numerical integration of Faraday's Law, assuming uniform corrosion and time-varying activity.

## Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/multiphysics-anode-model.git
   cd multiphysics-anode-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install numpy scipy pandas matplotlib seaborn
   ```

## Repository Structure

- `multiphysics_model.py`: Core implementation of JMAK and Electrochemical models, including parameter estimation and service life prediction.
- `uncertainty_analysis.py`: Advanced documentation for parameter uncertainty, including Bootstrap resampling and covariance analysis.
- `Multi-Physics_Supplementary_Materials_03032026.docx`: Detailed numerical results, statistical tables (S1-S15), and mathematical derivations.

## Usage

### Running the Model
To execute the main model fit and service life simulation:
```bash
python multiphysics_model.py
```

### Uncertainty Quantification
To perform bootstrap resampling and generate parameter distribution plots:
```bash
python uncertainty_analysis.py
```

## Methodology

- **Parameter Estimation**: Non-linear weighted least squares using the Trust Region Reflective (TRF) algorithm.
- **Uncertainty Quantification**: Non-parametric bootstrap resampling ($n=1000$) to determine 95% confidence intervals and parameter correlations.
- **Validation**: 5-fold cross-validation and comparison against independent experimental datasets.

## Citation

If you use this code in your research, please cite:
> *Research Team (2026). Predictive Multi-Physics Modelling of Al-Zn-Mg-Sn Sacrificial Anodes: A Data-Driven Microstructure-to-Performance Framework. [Journal Name/DOI pending]*

## License

Distributed under the MIT License. See `LICENSE` for more information.
