# AI for Cybersecurity in Automotive CAN Networks
__Intrusion Detection Systems (IDU) using Lightweight Machine Learning and Deep Learning__

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

### Background
Modern vehicles rely on CAN for ECU communication, but CAN lack built-in security since they are not encrypted and authenticated. This makes it (CAN) vunerable to attacks such as:
  - spoofing attacks
  - replay attacks
  - injection attack

Therefore, a lightweight IDS is critical for real detection and mitigation. 

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
```

### Install required dependencies
```bash
pip install -r requirements.txt
```

## Action
__Data Preprocessing__ 
To improve the data quality, the raw CAN logs were:
- parsed
- cleaned
- scaled/normalized
- attack labels were mapped to create a supervised learning problem.

__Feature Engineering__
To improve intrusion detection on the CAN bus, we created the following features.
- __Time-Based Features:__ Inter-message arrival times, frequency.
- __Payload-Level Features (for XGB):__ Statistical measures (mean, std) of data bytes.
- __Bit-Level Features:__ For entropy analysis.
- __DLC-Based Feature:__ Compliance checks for message length.
- __Sliding-Window:__ To create sequence for Deep Learning Models-LSTM/CNN.
  
These features help to detect spoofing, replay, flooding, and abnormal message patterns that are not visible from raw data alone.

## Models Implemented
- **XGBoost**(engineered features): As a strong, interpretable baseline.
- __1D CNN:__ payload bytes extraction for local spatial patterns.
- __LSTM:__ temporal sequence learning and models the CAN bus as a time series.

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
