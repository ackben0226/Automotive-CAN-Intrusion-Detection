# AI for Cybersecurity in Automotive CAN Networks
__Intrusion Detection Systems (IDS) using Lightweight Machine Learning and Deep Learning__

## Research Contribution
This work is a comprehensive comparative analysis between sequence-based deep learning and tree-based machine learning for real-time Controller Area Network (CAN) intrusion detection under automotive-grade latency constraints. We demonstrate that Long Short-Term Memory (LSTM) networks achieve 91.56% accuracy with <10ms inference latency on ARM Cortex-M class processors, outperforming traditional methods while meeting the strict timing requirements of safety-critical automotive systems.

### Key Findings:
- LSTM models capture temporal attack patterns with 91.62% F1-score and 2.81% false positive rate
- XGBoost provides optimal accuracy-latency tradeoff (89.24% accuracy, <5ms) for resource-constrained ECUs
- Feature engineering (inter-arrival timing, payload entropy) improves detection by 8-12% across all models
- Real-time deployment feasible on automotive-grade hardware (NXP S32K series, 2MB Flash, 512KB RAM)

### Background
Modern vehicles contain 70-100 Electronic Control Units (ECUs) to communicate through the Controller Area Network (CAN) protocol. The CAN transmits and processes over 2000 messages per seconds between all the ECUs in typical driving scenarios. However, CAN lacks security mechanisms—messages such as encryption, authentication, or sender verification. This exposes critical vehicle functions, including braking, steering, powertrain, etc. to cyber attacks.

__Attack Landscape:__
- __Spoofing:__ Malicious ECUs impersonate legitimate controllers
- __Replay:__ Captured legitimate messages retransmitted out of context
- __Denial-of-Service (DoS):__ High-priority message flooding prevents critical communication
- __Fuzzing:__ Random data injection to trigger undefined ECU behaviour
<br/>  
This, however, demonstrates the urgent need for lightweight, real-time intrusion detection deployable on existing automotive hardware without redesigning CAN infrastructure.

## Dataset
This project uses the public [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) by the Hacking and Countermeasure Research Lab (HCRL). 
__Composition:__
- __Normal Traffic:__ 2,369,397 CAN messages (attack-free driving scenarios)
- __Attack Types:__
  - DoS: 656,578 messages
  - Fuzzy: 591,989 messages
  - Impersonation: 995,471 messages
    
- __Format:__ CAN frames with ID (11/29-bit), DLC (Data Length Code), DATA[0-7], 8-byte payload, timestamp, flags
- __Collection Environment:__ 2017 Hyundai Sonata, real-world driving conditions

## Dataset Statistics:
|Attack Types  | Messages | Percentage |
|:---:|:---:|:---:|
|__Normal Traffic__ |__2,369,397__ |__51.4%__ | 
| __DoS__  |__656,578__ | __14.2%__| 
| __Fuzzy__ | __591,989__ |__12.83%__ |
|__Impersonation__ | __995,471__ |__21.57%__|


## Model Performance Summary
__Attack Detection Results__

| Model | Accuracy |F1-Score | False Positive Rate|Latency |Model Size |
| :--- | :---: | :---: |:--:|:--:|:---|:--: |
| **LSTM** | **91.56%** | **91.62%**|**2.85%**|<10 ms | 2.3MB| 
| **XGBoost** | __89.24%__ |**89.04%** |**3.67%**|<5 ms| 1.1MB| 
| **1D CNN** | __84.00%__ | **82.98%**|**5.9%** |<10 ms|1.8MB | 


__Per-Attack Type Performance (LSTM)__  
| Class           | Precision | Recall | F1-score | Support |
|:---------------:|:---------:|:------:|:--------:|:-------:|
| Normal          | 0.9703    | 0.8884 | 0.9275   | 30000   |
| DoS             | 0.8473    | 0.8855 | 0.8660   | 29999   |
| Fuzzy           | 0.8550    | 0.8884 | 0.8714   | 30000   |
| Impersonation   | 0.9997    | 1.0000 | 0.9999   | 30000   |



## Methodology
### Data Preprocessing Pipeline
__Data Cleaning__
- Removed 1,247 messages with duplicate timestamps (0.03% of dataset)
- Validated DLC consistency (flagged 89 corrupted frames)
- Handled missing payload bytes with forward-fill strategy
- Normalized continuous features using RobustScaler (resilient to outliers)

__Label Encoding__
- Multi-class classification: {Normal: 0, DoS: 1, Fuzzy: 2, Impersonation: 3}
- Stratified train/validation/test split: 70/15/15
- Balanced sampling for minority classes during training

__Feature Engineering__
To detect sophisticated attacks invisible in raw CAN data, we engineered domain-specific features:
- __Temporal Features:__
- Inter-message arrival time: Detects DoS flooding patterns
- Message frequency per CAN ID: Identifies abnormal transmission rates
- Sliding window statistics (mean, std, min, max over 10-frame windows)

__Payload-Level Features:__
- Byte-wise statistics: mean, standard deviation, range across DATA[0-7]
- Shannon entropy: Measures payload randomness (high entropy → fuzzing attacks)
- Hamming distance: Compares consecutive payloads for replay detection

__Structural Features:__
- DLC compliance checking: Validates expected message lengths per CAN ID
- CAN ID transition matrix: Models legitimate communication patterns
- Bit-level analysis: Priority bit manipulation detection

__Sequence Representation:__
- Sliding window (50 frames) for LSTM/CNN temporal modeling
- Captures multi-message attack sequences (critical for impersonation detection)

## Model Architectures
### LSTM (Best Overall Performance)
```bash
Input: 10-frame sequences × 27 features
- LSTM Layer 1: 128 units, return_sequences=True
- Dropout: 0.3
- LSTM Layer 2: 64 units, return_sequences=False
- Dropout: 0.3
- Dense Layer 1: 128 units, ReLU activation
- Batch Normalization
- Dropout: 0.5
- Dense Layer 2: 64 units, ReLU activation
- Dropout: 0.4
- Output: 4 classes, Softmax
- Total Parameters: 146,728
```
### XGBoost (Best Latency-Accuracy Tradeoff)
```bash
- max_depth: 8
- n_estimators: 200
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- Features: 38 engineered features (no sequences)
```
### 1D CNN (Payload Pattern Extraction)
```bash
Input: 50-frame sequences × 8 payload bytes
- Conv1D: 64 filters, kernel=3, ReLU
- MaxPooling1D: pool_size=2
- Conv1D: 32 filters, kernel=3, ReLU
- GlobalAveragePooling1D
- Dense: 64 units, ReLU
- Output: 4 classes, Softmax

# 1D CNN Architecture

Input: 27 timesteps × 1 channel

# --- Convolutional Feature Extractor ---
- Conv1D: 64 filters, kernel_size=5, activation=ReLU, padding="same"
- BatchNormalization
- Conv1D: 64 filters, kernel_size=5, activation=ReLU, padding="same"
- BatchNormalization
- MaxPooling1D: pool_size=2
- Dropout: 0.3

- Conv1D: 128 filters, kernel_size=3, activation=ReLU, padding="same"
- BatchNormalization
- Conv1D: 128 filters, kernel_size=3, activation=ReLU, padding="same"
- BatchNormalization
- MaxPooling1D: pool_size=2
- Dropout: 0.3

- GlobalAveragePooling1D
- Dense: 128 units, activation=ReLU, L2 regularization=1e-4
- Dropout: 0.4
- Dense: 64 units, activation=ReLU, L2 regularization=1e-4
- Dropout: 0.3
- Output Layer: Dense, 4 units, activation=Softmax

# Parameter Summary
Total Parameters: 121,476
Trainable Parameters: 120,708
Non-trainable Parameters: 768
```

## Installation
### Prerequisites
- Python 3.8 and higher
- pip or conda

### Setup
```bash
# Clone repository
git clone https://github.com/ackben0226/Automotive-CAN-Intrusion-Detection.git
cd Automotive-CAN-Intrusion-Detection

# Create virtual environment
conda create -n can_env python=3.11 (Anaconda prompt)
```

### Install required dependencies
```bash
pip install -r requirements.txt

# Download dataset (automated script)
python scripts/download_dataset.py --output data/raw/

# Verify installation
python scripts/verify_setup.py
```
  
## Result & Discussion
The best performance of the LSTM model demonstrates the importance of **temporal context** in CAN intrusion detection. It has the ability to remember past message pattern, which is crucial for identifying sophisticated multi-frame attacks like impersonation.
*   **XGBoost** offered an excellent trade-off between accuracy (89.24%) and very low latency (<5ms), making it a prime candidate for the most resource-limited ECUs.
*   The **CNN** model was effective at spotting anomalous payload structures but was slightly less accurate than the sequence-aware LSTM.

All models are sufficiently lightweight for real-time inference on embedded hardware.

## Project Structure
```bash
Automotive-CAN-Intrusion-Detection/
├── config/                     # Configuration files (YAML)
│   ├── model_config.yaml
│   ├── preprocess_config.yaml
│   └── train_config.yaml
├── data/                       
│   ├── raw/                    # Original HCRL dataset
│   ├── processed/              # Preprocessed features & sequences
│   └── download_dataset.py     # Automated download script
├── notebooks/                  
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Comparison.ipynb
├── src/
│   ├── preprocessing/          
│   │   ├── clean_data.py
│   │   └── preprocess_data.py
│   ├── features/              
│   │   ├── temporal_features.py
│   │   ├── payload_features.py
│   │   └── sequence_generator.py
│   ├── models/                
│   │   ├── lstm.py            # LSTM architecture
│   │   ├── xgboost_model.py   # XGBoost wrapper
│   │   ├── cnn.py             # 1D CNN architecture
│   │   ├── train_lstm.py
│   │   ├── train_xgboost.py
│   │   └── train_cnn.py
│   └── evaluation/            
│       ├── evaluate.py        # Metrics calculation
│       ├── benchmark_latency.py
│       └── visualize_results.py
├── results/                   
│   ├── confusion_matrix.png
│   ├── latency_distribution.png
│   └── training_curves.png
├── checkpoints/               # Saved model weights
├── requirements.txt
├── README.md
└── LICENSE
```

## Limitations & Future Work
### Limitations
- Model trained on a single dataset (2017 Hyundai Sonata) — cross-vehicle evaluation needed.
- Cold-Start Problem: First 50 messages show 34% false positive rate. Mitigation: Pre-load startup sequences.
- Novel Attack Generalization: 67% detection on unseen patterns. Future: Hybrid anomaly + supervised.
- Timing-Based Attack Evasion: 78% detection on sophisticated timing mimics. Requires side-channel features.

## Extension to Time-Sensitive Networking (TSN) and Next-Generation Automotive Cybersecurity
This work is built on the Controller Area Network (CAN), and remains widely used in production vehicles. The methods developed, particularly timing-based feature extraction and low-latency models are not tied to CAN only. They can be applied to TSN-based in-vehicle networks, including those studied in the TITAN project at the University of Luxembourg.

### Key Transferable Elements
- **Timing-based features**: Inter-message gaps, message rates, and sliding-window statistics are effective for detecting DoS and replay attacks on CAN. In TSN, the same signals can expose schedule violations, jitter, or delay manipulation that interfere with time-aware shaping (IEEE 802.1Qbv) and clock synchronization (IEEE 802.1AS).
  
- **Low-latency models**: The LSTM (91.6% accuracy, <10 ms inference) and XGBoost (<5 ms inference) models were designed for microcontroller-class hardware (e.g., ARM Cortex-M, NXP S32K). These constraints are comparable to those of TSN gateways and ECUs, where detection must not interfere with real-time control tasks.

- **Threat Model Alignment**:The detected attack classes—spoofing, replay, DoS, and fuzzing—map directly to risks addressed in ISO/SAE 21434 and UNECE WP.29. In TSN or TSN–SDN systems, equivalent attacks can target scheduled streams or control-plane updates, leading to the same safety and compliance failures.


### Direction for PhD Research (TITAN Context)
This CAN-based system provides a practical base for work on TSN security (AI-enhanced cybersecurity):
- Evaluate the models on TSN traffic using OMNeT++/INET, focusing on time-critical streams rather than aggregate throughput.
- Study adaptive responses using reinforcement learning, such as adjusting schedules or isolating flows once timing anomalies are detected.
- Extend detection to TSN-specific failure modes, including clock desynchronization and reservation manipulation, using hybrid supervised/unsupervised methods.
- Validate results in industrial testbeds during secondments, with emphasis on standards compliance rather than benchmark accuracy.

This work demonstrates hands-on expertise in applied ML for automotive IDS and a clear vision for advancing threat-aware systems in TSN environments.

### Future Research Directions
- __Federated Learning:__ Aggregate knowledge across vehicle fleets without sharing raw CAN data
- __Zero-Shot Detection:__ Generative models (VAE, GAN) for detecting never-before-seen attacks
- **Multi-Modal Fusion:** Combine CAN traffic with sensor data (accelerometer, GPS) for context-aware detection
- **Adversarial Robustness:** Defense against adaptive attackers aware of IDS model
- **Hardware Security Module (HSM) Integration:** Cryptographic authentication alongside ML-based detection

## Visual Results
### Confusion Matrix (LSTM Model)
<img src="https://github.com/user-attachments/assets/01d632a6-c99c-4ee1-b25c-e6e39522a20b"
     alt="Confusion Matrix"
     width="760"
     height="590">

### Latency Distribution
<img width="695" height="545" alt="image" src="https://github.com/user-attachments/assets/bd1d68a7-25a3-4125-b260-3b8ea942c3b9" />

### Training Curves
<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/4060a487-350a-47a7-a556-4dbe25e58a31" />


## References
- H. Lee et al., "CAN ID Shuffling Technique (CIST): Moving Target Defense for the In-Vehicle Network," 2021.
- HCR Lab, "Car Hacking Dataset," IEEE Dataport, 2020.
