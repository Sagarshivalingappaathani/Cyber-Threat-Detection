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

- **Fuzzers:**  
  These attacks send random or unexpected data to a system to see how it reacts and find weaknesses.  
  _Example:_ Imagine a website’s contact form. A fuzzer attack would enter strange symbols, extremely long text, or unexpected data types to crash the website or make it behave oddly — revealing vulnerabilities developers didn’t expect.  

- **Analysis:**  
  Scanning a network to gather information and identify weak points.  
  _Example:_ A hacker uses a tool like **Nmap** to scan a company’s network and see which devices are connected and which ports are open. Open ports can sometimes be entry points for attacks.  

- **Backdoors:**  
  Secret, unauthorized ways to access a system without going through normal security.  
  _Example:_ A hacker installs a hidden program on a computer that creates a secret login route. This lets them enter the system without needing a username or password, even after security updates are applied.  

- **DoS (Denial of Service):**  
  Overwhelming a website or network with too much traffic, making it slow or unavailable.  
  _Example:_ An online shopping website gets flooded with millions of fake requests at once, making the site so slow that real customers can’t access it. This is like hundreds of people blocking a store’s entrance so no one else can get inside.

- **Generic:**  
  Attacks on encrypted data by trying every possible password or key until one works.  
  _Example:_ A hacker uses a **brute-force attack** on an email account by trying thousands of common passwords until they find the right one. It’s like trying every key on a keychain until one unlocks the door. 

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
 
 


![Alt text](xgboost_performance.png)
![Alt text](random_forest_performance.png)
