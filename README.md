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


## Project Overview
In this project, we apply machine learning and deep learning to implement __lightweight intrusion detection system (IDS)__ for automotive Controller Area Network (CAN) traffic. The idea behind this project is to detect malicious CAN messages in real time on resource-constrained Electronic Control Units (ECUs).

## Model Performance Summary
To achieve this, we compared three models and their accuracies are as follow: 

| Model | Accuracy | Latency | Best For |
| :--- | :---: | :---: | :--- |
| **LSTM** | **91.56%** | <10 ms | Learning temporal sequences and long-range dependencies |
| **XGBoost** | __89.24%__ | <5 ms | Tabular data with engineered features |
| **1D CNN** | __84.00%__ | <10 ms | Extracting local payload patterns |

After training the model, the LSTM model achieved the highest performance and maintained __<10ms__ __latency__. This makes it suitable for embedded automotive environments.

### Dataset
This project uses the public __Car-Hacking-Dataset__ by the Hacking and Countermeasure Research Lab (HCRL). The dataset has the following contents:
- __Normal driving data (Attack-free):__ 2,369,397 CAN messages
- __Attack types:__ DoS, Fuzzy and Impersonation
- __Format:__ CAN frame data (ID, DLC, DATA[0-7], Timestamp, Load, Flag)

__The dataset for this project is hosted on Download dataset:__ 

## Installation
### Prerequisites
- Python 3.8+
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
```

## Action
__Data Preprocessing__ 
<br/>To improve the data quality and model performance, the raw CAN logs were:
1. __Parsed & Cleaned__ to:
   - remove duplicate timestamps
   - handle missing/corrupted frames
   - validated DLC (Data Length Code) consistency
     
2. __scaled/normalized__ for continuous features
3. __map labeled__  for the attack, Normal, DoS, Fuzzy, Impersonation, to create a supervised learning problem.

__Feature Engineering__
<br/>To improve intrusion detection on the CAN bus, the following features were engineered.
- __Time-Based Features:__ Inter-message arrival times, frequency.
- __Payload-Level Features (for XGB):__ Statistical measures (mean, std) of data bytes.
- __Bit-Level Features:__ For entropy analysis.
- __DLC-Based Feature:__ Compliance checks for message length.
- __Sliding-Window:__ To create sequence for Deep Learning Models-LSTM/CNN.
  
These features help to detect spoofing, replay, flooding, and abnormal message patterns that are not visible from raw data alone.

## Models Implemented
- **XGBoost**(engineered features): As a strong, interpretable baseline and efficiently learns non-linear decision boundaries
- __1D CNN:__ payload bytes extraction for local spatial patterns. It filters detect malicious payload patterns
- __LSTM:__ temporal sequence learning and models the CAN bus as a time series. LSTM captures patterns and flags deviations.

## Result & Discussion
The best performance of the LSTM model demonstrates the importance of **temporal context** in CAN intrusion detection. It has the ability to remember past message pattern, which is crucial for identifying sophisticated multi-frame attacks like impersonation.
*   **XGBoost** offered an excellent trade-off between accuracy (89.24%) and very low latency (<5ms), making it a prime candidate for the most resource-limited ECUs.
*   The **CNN** model was effective at spotting anomalous payload structures but was slightly less accurate than the sequence-aware LSTM.

All models are sufficiently lightweight for real-time inference on embedded hardware.

## Project Structure
```bash
Automotive-CAN-Intrusion-Detection/
├── data/ # Scripts for downloading & preprocessing dataset
├── notebooks/ # Jupyter notebooks for EDA and prototyping
├── src/
│ ├── features/ # Feature engineering pipelines
│ ├── models/ # XGBoost, CNN, and LSTM model definitions & training scripts
│ └── evaluation/ # Scripts for model evaluation & latency testing
├── requirements.txt
└── README.md
```

## References
- H. Lee et al., "CAN ID Shuffling Technique (CIST): Moving Target Defense for the In-Vehicle Network," 2021.
- HCR Lab, "Car Hacking Dataset," IEEE Dataport, 2020.
