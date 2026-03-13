# WiFi-Based Human Presence Detection Using CSI and LSTM

A human activity detection system that uses **WiFi Channel State Information (CSI)** captured from an ESP32-C5 module and classifies room occupancy into three states using a **Long Short-Term Memory (LSTM)** neural network.

| Class | Label | Description |
|-------|-------|-------------|
| 0 | No Presence | No human detected inside the room |
| 1 | Human Presence (Idle) | A person is present with minimal or no movement |
| 2 | Movement | A person is actively moving within the room |

## Introduction

Wireless sensing is an emerging technology that utilizes wireless signals to detect and monitor surrounding objects by analyzing changes in radio signals as they propagate through an environment. When a person moves, their body interacts with the wireless signal, causing measurable variations.

**Channel State Information (CSI)** describes how a WiFi signal changes while traveling from the transmitter to the receiver. Unlike simple signal strength (RSSI), CSI provides fine-grained, subcarrier-level amplitude and phase information across 53 subcarriers — forming a rich time-series that is highly sensitive to human presence and movement.

This project collects CSI data using an **ESP32-C5** WiFi module, preprocesses it into fixed-length windows, and trains an LSTM classifier to detect human activity in a room.

## Project Architecture

```
Step 1: Data Collection (ESP32-C5 + WiFi Router)
        ↓
Step 2: Data Preprocessing (Feature Extraction → Window Slicing → Denoising)
        ↓
Step 3: Data Splitting (GroupShuffleSplit — 85% TEMP / 15% TEST)
        ↓
Step 4: Model Development (LSTM + StratifiedGroupKFold Cross-Validation)
        ↓
Step 5: Evaluation (Held-out TEST set — Confusion Matrix & Classification Report)
```
![image alt](https://github.com/teetat9/CSI_Detection/blob/95cd5dbfc9d1b86f94fd8a610bbefb3905cc627a/Screenshot%202026-03-13%20144859.png)

## Repository Structure

```
CSI_Detection/
├── data_collection/          # Scripts and firmware for ESP32-C5 CSI data collection
├── csi_dataset/              # Preprocessing notebooks and the final dataset (.pt)
│   └── csi_windows_w32_s16.pt   # Preprocessed dataset (X, y, groups)
├── lstm/                     # LSTM training notebooks and saved model checkpoints
├── environment.yml           # Conda environment specification
├── LICENSE
└── README.md
```

## Dataset

Data was collected using an ESP32-C5 module and a WiFi router in a controlled room environment.

- **5 recordings per class**, each 30 seconds long → **15 CSV files** total
- Each CSV row contains CSI amplitude values from **53 WiFi subcarriers**
- After window slicing: **173 windows** of shape `(32 timesteps, 53 features)`
- Each window is assigned a **group ID** (the original recording file it came from) to prevent data leakage during splitting

### Preprocessing Pipeline

1. **Feature Selection** — Extract the 53 amplitude subcarrier columns as features (X) and the label column as target (y)
2. **Window Slicing** — Slide a fixed-size window (size=32, stride=16) over each recording to create input sequences for the LSTM
3. **Denoising** — Handle NaN/Inf values, apply interpolation for temporal continuity, and clip extreme values using quantile clipping (lower=0.001, upper=0.999)

## Data Splitting Strategy

Sliding windows from the same recording overlap in time (e.g., window 1 covers t1–t32, window 2 covers t17–t48). Randomly splitting these across train and test sets would cause **data leakage** — the model would see nearly identical data in both sets.

To prevent this:

1. **GroupShuffleSplit** splits the data into TEMP (85%) and TEST (15%), keeping all windows from the same recording together
2. A random state search (0–200) finds a seed where the TEST set contains all 3 classes (`random_state=2`)
3. Within TEMP, **StratifiedGroupKFold** is used for cross-validation, again respecting group boundaries

## Model

### LSTM Architecture

```
Input (B, 32, 53)  →  LSTM layers  →  Last timestep (B, hidden_size)  →  Linear (B, 3)
```

- **Input**: batch of CSI windows, shape `(batch_size, 32, 53)`
- **LSTM**: encodes the full time-series, output shape `(B, 32, hidden_size)`
- **Last timestep**: only the final hidden state `out[:, -1, :]` is used as the sequence summary
- **Fully connected layer**: maps to 3 class logits

### Training Details

- **Loss**: `CrossEntropyLoss` with balanced class weights (computed from training labels only) to handle class imbalance
- **Optimizer**: Adam
- **Normalization**: Z-score normalization fitted on training data only, then applied to both train and validation/test
- **Cross-validation**: StratifiedGroupKFold (K=3 for sanity check, K=5 for hyperparameter search)

### Hyperparameter Search

The following hyperparameters were tuned via cross-validation:

| Parameter | Values Tested |
|-----------|---------------|
| Hidden Size | 64, 128 |
| Number of Layers | 1, 2 |
| Dropout | 0.0, 0.3 (only when layers > 1) |
| Learning Rate | 1e-3, 5e-4 |

**Best configuration**: hidden_size=64, num_layers=1, dropout=0.0, learning_rate=1e-3

### Results

| Model Config | Mean Validation Accuracy |
|-------------|--------------------------|
| h64_l1 | **0.9867** |
| h64_l2 | 0.9591 |
| h128_l1 | 0.9864 |
| h128_l2 | 0.9793 |

**Final TEST accuracy: 97.22%** (on the held-out 15% test set, evaluated once)

## How to Run

### 1. Set Up the Environment

```bash
conda env create -f environment.yml
conda activate <env_name>
```

Or install the core dependencies manually:

```bash
pip install torch numpy scikit-learn matplotlib
```

### 2. Data Collection (Optional — Pre-collected Data Included)

If you want to collect your own CSI data:

1. Set up an ESP32-C5 module with the [Espressif CSI firmware](https://github.com/espressif/esp-csi/tree/master)
2. Place the ESP32-C5 and a WiFi router in a room
3. Record 30-second sessions for each class (no presence, idle, movement)
4. Save the output as CSV files in `data_collection/`

### 3. Data Preprocessing

Open and run the preprocessing notebook in `csi_dataset/`:

```bash
cd csi_dataset
jupyter notebook
```

This notebook will:
- Load raw CSV files
- Extract amplitude features from 53 subcarriers
- Apply window slicing (window=32, stride=16)
- Apply denoising filters
- Save the final dataset as `csi_windows_w32_s16.pt`

### 4. Train and Evaluate the LSTM

Open and run the training notebook in `lstm/`:

```bash
cd lstm
jupyter notebook
```

The notebook performs the following steps in order:

1. **Load dataset** — reads `csi_dataset/csi_windows_w32_s16.pt`
2. **Split data** — GroupShuffleSplit into 85% TEMP and 15% TEST
3. **Sanity check** — 3-fold cross-validation with default hyperparameters
4. **Hyperparameter search** — 3-fold cross-validation over the parameter grid
5. **Final training** — train on all TEMP data with the best config
6. **Evaluation** — evaluate once on the held-out TEST set (confusion matrix + classification report)
7. **Save model** — exports the trained model to `lstm/csi_lstm_group85_test15.pt`

### 5. Load a Saved Model (for Inference)

```python
import torch
from model import LSTMClassifier  # or define the class as in the notebook

checkpoint = torch.load("lstm/csi_lstm_group85_test15.pt", map_location="cpu")

model = LSTMClassifier(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    num_classes=checkpoint["num_classes"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Reference

- [Espressif ESP-CSI](https://github.com/espressif/esp-csi/tree/master) — The data collection code used in this project is based on the official Espressif ESP-CSI toolkit for extracting Channel State Information from ESP32 devices.

## Authors

- **Thitat Sawamipak** — 67110032
- **Sarawit Phairo** — 67011296

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
