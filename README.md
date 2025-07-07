
# AetherAI: Advanced Air Quality Prediction System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

AetherAI is an advanced deep learning system that utilizes Long Short-Term Memory (LSTM) neural networks for multi-step air quality forecasting. The project focuses on predicting atmospheric conditions, particularly temperature, using historical environmental data through sophisticated time-series analysis.

## 🎯 Project Overview

## 🎯 Project Overview

AetherAI addresses the critical challenge of environmental monitoring and prediction by developing robust time-series forecasting models. The system can accurately predict temperature variations for the next 'n' hours based on comprehensive historical air quality datasets.

### Key Features

- **Dual Architecture Approach**: Implements both Vector Output and Encoder-Decoder methodologies
- **Multi-step Forecasting**: Predicts multiple future time steps simultaneously
- **Data Leakage Prevention**: Ensures model integrity through proper validation techniques
- **Comprehensive Preprocessing**: Advanced data cleaning and feature engineering pipeline
- **Visualization Tools**: Interactive plots for model performance analysis

### Forecasting Methods

1.  **Vector Output Method**: A streamlined multivariate model that outputs a single prediction vector for future timesteps
2.  **Encoder-Decoder Method**: A sophisticated architecture enabling parallel sequence-to-sequence predictions

## 📊 Dataset

## 📊 Dataset

The model utilizes the comprehensive "Air Quality" dataset containing multi-dimensional atmospheric measurements. The dataset includes various environmental parameters that influence air quality conditions.

### Data Preprocessing Pipeline

- **Quality Control**: Systematic removal of invalid data points (anomalous -200 values)
- **Feature Selection**: Strategic column elimination to optimize model performance
- **Normalization**: StandardScaler implementation for numerical stability and convergence
- **Categorical Encoding**: LabelEncoder transformation for temporal features
- **Missing Value Handling**: Robust imputation strategies for incomplete records

### Data Characteristics

- **Temporal Resolution**: Hourly measurements
- **Feature Space**: Multi-variate environmental parameters
- **Target Variable**: Temperature (T) with continuous values
- **Train/Test Split**: 75%/25% with temporal integrity preservation

## 🔬 Methodology

## 🔬 Methodology

AetherAI implements cutting-edge deep learning architectures specifically designed for time-series forecasting. The project explores two complementary approaches to multi-step prediction:

### 🏗️ Model Architectures

#### 1. Vector Output Model (Direct Multi-Output)

**Architecture Details:**
- **Input Layer**: Sequences of 1024 timesteps with multiple features
- **LSTM Layers**: 
  - Layer 1: 512 units with 40% dropout and return_sequences=True
  - Layer 2: 256 units with 40% dropout and return_sequences=True  
  - Layer 3: 128 units with 40% dropout
- **Output Layer**: Dense layer producing 24 future predictions
- **Training Strategy**: Direct optimization with learning rate scheduling

**Advantages:**
- Computationally efficient for single-target prediction
- Direct end-to-end training
- Suitable for temperature-specific forecasting

#### 2. Encoder-Decoder Model (Sequence-to-Sequence)

**Architecture Details:**
- **Encoder**: 256-unit LSTM with 40% dropout (512 timesteps input)
- **RepeatVector**: Context distribution across output timesteps
- **Decoder**: 256-unit LSTM with return_sequences=True
- **Output Layer**: TimeDistributed Dense for parallel multi-feature prediction
- **Training Strategy**: Multi-objective optimization across all environmental parameters

**Advantages:**
- Handles multiple output features simultaneously
- Better context preservation for complex patterns
- Scalable to additional environmental variables

### 🎛️ Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with adaptive learning rate
- **Learning Rate Schedule**: Exponential decay every 22/50 epochs
- **Batch Sizes**: 64 (Vector Output) / 128 (Encoder-Decoder)
- **Epochs**: 70 / 200 respectively
- **Regularization**: Dropout layers for overfitting prevention

## 📈 Results & Performance

## 📈 Results & Performance

The project demonstrates robust forecasting capabilities through comprehensive evaluation metrics and visualization analysis.

### Model Evaluation

- **Validation Strategy**: Time-series cross-validation without data leakage
- **Prediction Horizon**: 24-hour ahead forecasting
- **Performance Metrics**: Mean Squared Error (MSE) minimization
- **Visualization**: Comparative plots showing train/test/prediction trajectories

### Key Achievements

- **Multi-step Accuracy**: Successful prediction of temperature sequences
- **Pattern Recognition**: Model captures seasonal and temporal variations
- **Generalization**: Robust performance on unseen test data
- **Scalability**: Architecture supports extended prediction horizons

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed with the following dependencies:

```bash
pip install tensorflow>=2.0
pip install keras
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install tqdm
```

### Installation & Usage

1. **Clone or Download** the project files
2. **Prepare Dataset**: Ensure `Air Quality.csv` is available in the specified path
3. **Launch Jupyter**: Open `final-lstm-model-9c884d (1).ipynb` in your preferred environment
4. **Execute Sequentially**: Run all cells in order for complete analysis
5. **Customize Parameters**: Modify `n_steps_in`, `n_steps_out`, and architecture settings as needed

### Quick Start Example

```python
# Load and preprocess data
df = pd.read_csv("Air Quality.csv")
# ... preprocessing steps ...

# Create sliding windows
train_X, train_y = sliding_window(train_df, n_steps_in=1024, n_steps_out=24)

# Build and train model
model = Sequential([
    LSTM(512, return_sequences=True, dropout=0.4),
    LSTM(256, return_sequences=True, dropout=0.4),
    LSTM(128, dropout=0.4),
    Dense(24)
])
model.compile(optimizer='Adam', loss='mse')
model.fit(train_X, train_y, epochs=70)
```

## 📁 Project Structure

```
AetherAI/
├── final-lstm-model-9c884d (1).ipynb    # Main analysis notebook
├── README.md                             # Project documentation
├── Air Quality.csv                       # Dataset (not included)
└── requirements.txt                      # Dependencies (optional)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Model architecture improvements
- Additional evaluation metrics
- Performance optimization
- Documentation enhancements
- New forecasting approaches

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- Long Short-Term Memory Networks for Time Series Forecasting
- TensorFlow/Keras Documentation
- Air Quality Monitoring and Prediction Techniques
- Deep Learning for Environmental Data Analysis

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.
