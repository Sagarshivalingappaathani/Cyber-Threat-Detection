# Cyber-Threat-Detection
## About Dataset
The [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a widely used dataset for evaluating network intrusion detection systems (NIDS). It was created by the **Australian Centre for Cyber Security (ACCS)** and is designed to address the limitations of older datasets like **KDDCup99** and **NSL-KDD** by providing a more realistic and diverse representation of modern network traffic and attack behaviors.  

## Dataset Composition

- **Total Records:** 2,540,044  
- **Attack Types:** 9 different types of attacks  
- **Features:** 49 features divided into:  
  - **Basic Features:** Packet and protocol-level attributes  
  - **Content Features:** Payload information  
  - **Time-based Features:** Timing of packets and connections  
  - **Connection Features:** Aggregated flow-level characteristics  

## 9 Types Attacks

- **DoS (Denial of Service):**  
  Overwhelming a website or network with too much traffic, making it slow or unavailable.  
  _Example:_ An online shopping website gets flooded with millions of fake requests at once, making the site so slow that real customers can’t access it. This is like hundreds of people blocking a store’s entrance so no one else can get inside.

- **Fuzzers:**  
  These attacks send random or unexpected data to a system to see how it reacts and find weaknesses.  
  _Example:_ Imagine a website’s contact form. A fuzzer attack would enter strange symbols, extremely long text, or unexpected data types to crash the website or make it behave oddly — revealing vulnerabilities developers didn’t expect.  

- **Generic:**  
  Attacks on encrypted data by trying every possible password or key until one works.  
  _Example:_ A hacker uses a **brute-force attack** on an email account by trying thousands of common passwords until they find the right one. It’s like trying every key on a keychain until one unlocks the door.

- **Analysis:**  
  Scanning a network to gather information and identify weak points.  
  _Example:_ A hacker uses a tool like **Nmap** to scan a company’s network and see which devices are connected and which ports are open. Open ports can sometimes be entry points for attacks.  

- **Backdoors:**  
  Secret, unauthorized ways to access a system without going through normal security.  
  _Example:_ A hacker installs a hidden program on a computer that creates a secret login route. This lets them enter the system without needing a username or password, even after security updates are applied.  

- **Exploits:**  
  Taking advantage of bugs or flaws in software to gain control or cause damage.  
  _Example:_ An attacker finds a bug in an older version of a web application that lets them bypass login checks and access admin controls without a password. This is similar to finding a broken lock on a door and sneaking inside.  
 
- **Reconnaissance:**  
  Spying on a network to gather information before launching an attack.  
  _Example:_ Using **Wireshark**, a hacker monitors data flowing through a public Wi-Fi network at a coffee shop, capturing unencrypted information like usernames and passwords people enter on websites.  

- **Shellcode:**  
  Malicious code designed to take over a computer and execute commands remotely.  
  _Example:_ An attacker sends an infected PDF to a company’s employee. When the file is opened, hidden shellcode runs in the background, giving the attacker remote access to the employee’s computer.  

- **Worms:**  
  A type of virus that spreads from one computer to another on its own, without anyone clicking or opening anything.  
  _Example:_ The **WannaCry** worm spread to thousands of computers by taking advantage of a weakness in Windows. It locked people’s files and asked for money to unlock them.  


# Attack Detection Model Performance

## 1. Model Comparison for Attack Prediction (Binary Classification)
This table shows the performance of various models in predicting whether a connection is an attack or normal.

| Model                      | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) | Time to Train (s) | Time to Predict (s) | Total Time (s) |
|---------------------------|--------------|------------|----------------|--------------|------------------|--------------------|----------------|
| Logistic                   | 92.80        | 92.80      | 92.84          | 92.81        | 4.3               | 0.0                 | 4.3            |
| kNN                        | 95.04        | 95.04      | 95.09          | 95.05        | 0.0               | 2.5                 | 2.5            |
| Decision Tree              | 96.54        | 96.54      | 96.54          | 96.54        | 1.2               | 0.0                 | 1.2            |
| Random Forest              | 97.67        | 97.67      | 97.68          | 97.67        | 5.3               | 0.2                 | 5.5            |
| Gradient Boosting Classifier| 95.80       | 95.80      | 95.80          | 95.80        | 55.1              | 0.0                 | 55.1           |
| XGBoost Classifier         | 97.80        | 97.80      | 97.81          | 97.80        | 2.0               | 0.0                 | 2.0            |
| AdaBoost Classifier        | 91.97        | 91.97      | 91.98          | 91.97        | 7.9               | 0.1                 | 8.0            |
| MLP                        | 96.34        | 96.34      | 96.39          | 96.34        | 28.1              | 0.0                 | 28.1           |
| MLP (Keras)                | 96.21        | 96.21      | 96.21          | 96.21        | 19.6              | 0.7                 | 20.4           |

---

## 2. XGBoost Model Performance on Individual Attack Types (Binary Classification)
This table shows the performance of 9 separate binary classification models, each trained to detect a specific attack type versus all other traffic.


| Attack Category   | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) |
|------------------|--------------|------------|----------------|--------------|
| DOS               | 95.54        | 97.86      | 93.53          | 95.64        |
| Fuzzers           | 97.72        | 98.83      | 96.68          | 97.74        |
| Generic           | 99.48        | 99.29      | 99.67          | 99.48        |
| Analysis          | 97.45        | 99.60      | 95.50          | 97.50        |
| Reconnaissance    | 98.64        | 98.18      | 99.10          | 98.64        |
| Backdoor          | 97.96        | 99.88      | 96.19          | 98.00        |
| Shellcode         | 99.78        | 99.96      | 99.60          | 99.78        |
| Worms             | 99.96        | 99.99      | 99.93          | 99.96        |
| Exploit           | 94.84        | 99.52      | 91.00          | 95.07        |

---

## 2. Random Forest Performance on Individual Attack Types (Binary Classification)
This table shows the performance of 9 separate binary classification models, each trained to detect a specific attack type versus all other traffic.

| Attack Category   | Accuracy (%) | Recall (%) | Precision (%) | F1-Score (%) |
|------------------|--------------|------------|----------------|--------------|
| DOS               | 94.55        | 99.29      | 90.70          | 94.80        |
| Fuzzers           | 97.69        | 98.81      | 96.65          | 97.72        |
| Generic           | 99.50        | 99.16      | 99.83          | 99.49        |
| Analysis          | 97.52        | 99.68      | 95.55          | 97.57        |
| Reconnaissance    | 98.55        | 98.09      | 99.01          | 98.55        |
| Backdoor          | 98.02        | 99.90      | 96.27          | 98.05        |
| Shellcode         | 99.81        | 99.99      | 99.64          | 99.81        |
| Worms             | 99.98        | 99.98      | 99.98          | 99.98        |
| Exploit           | 94.54        | 99.27      | 90.70          | 94.79        |
