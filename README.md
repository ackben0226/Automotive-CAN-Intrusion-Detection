# AI for Cybersecurity in Automotive CAN Networks
__Intrusion Detection Systems (IDU) using Lightweight Machine Learning and Deep Learning__

## Project Overview
In this project, we apply machine learning and deep learning to implement __lightweight intrusion detection system (IDS)__ for automotive Controller Area Network (CAN) traffic. The idea behind this project is to detect malicious CAN messages in real time on resource-constrained Electronic Control Units (ECUs).

To achieve this, we developed three models and compared their accuracies. 
|Model|LSTM|XGBoost|CNN (1D)|
|-----|:----:|:----:|:----:|
|__Accuracy__|__91.56%__|__89%__|__84%__|

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

## Action
__Data Preprocessing__ 
To improve the data quality, the raw CAN dataset was:

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

- parsed
- cleaned
- scaled/normalized

__Feature Engineering__

To improve intrusion detection on the CAN bus, we created the following features.
- Time-Based Features
- Payload-Level Features (for XGB)
- Bit-Level Features
- DLC-Based Feature
- Sliding-Window Sequences (for Deep Learning Models)
  
These features help to detect spoofing, replay, flooding, and abnormal message patterns that are not visible from raw data alone.

## Models Implemented
- **XGBoost**(engineered features)
- __1D CNN__(payload pattern extraction)
- __LSTM__(temporal sequence learning)

## Result
__Model Perfomace__
|Model|Accuracy|Noted|
|LSTM|__91.56%__|Best for sequencial patterns|
|XGBoost|__89.24%__|Strong tabular baseline|
|:---:|:---:|:---:|


