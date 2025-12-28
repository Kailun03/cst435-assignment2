# CST435 Assignment 2: Parallel Image Processing on GCP

## 📖 Assignment Overview
This assignment will implement a parallel image processing system to benchmark different parallel 
programming paradigms. The programming language chosen in this assignment is **Python**.
The system applies a pipeline of five image filters to a dataset of food images and 
deployed on a **Google Cloud Platform (GCP)** Virtual Machine.

The primary goal is to analyze the performance trade-offs, scalability, and speedup between:
1.  **Multiprocessing** (Process-based parallelism using `multiprocessing`)
2.  **Concurrent Futures** (High-level asynchronous execution using `concurrent.futures`)

## 🎯 Objectives
* Implement a sequential image processing pipeline with 5 image filters.
* Parallelize the pipeline using two different Python paradigms.
* Deploy and test the solution on a controlled GCP environment.
* Analyze Speedup (Amdahl's Law) and Efficiency across different core counts (1, 2, 4, 8).

---

## ⚙️ Environment Setup

### 1. Google Cloud Platform (GCP) Instance
The project was developed and tested on the following GCP VM configuration:
* **Machine Type:** `e2-standard-4` (4 vCPUs, 16 GB Memory)
* **Operating System:** Debian GNU/Linux 12 (bookworm)
* **Region:** asia-southeast1 (Singapore)
* **Disk:** 10 GB Balanced Persistent Disk

### 2. System Dependencies
To set up the environment on a fresh VM, execute:

```
# Update system and install system-level dependencies for OpenCV
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv unzip libgl1
```

### 2. Virtual Environment
create and activate a Python virtual environment:

```
#Isolate project dependencies by creating a virtual environment named venv
$ python3 -m venv venv

# Activate the environment
$ source venv/bin/activate
```

## 📂 Dataset Preparation
This project uses a subset of the Food-101 dataset.
To avoid storage issues on the VM, we utilize a manual subset of 100 images (specifically images from the apple_pie class).

### Step 1: Upload Zip Folder
Use the "Upload File" utility to upload `cst435-dataset.zip` to the GCP VM via SSH.

### Step 2: Extraction
Run the following commands in the SSH terminal to unzip the dataset into a clean directory:
```
# Create a target directory
mkdir image-data

# Unzip the dataset
unzip cst435-dataset.zip -d image-data
```

## 📸 Image Processing Pipeline
The system applies the following filters to every image in the dataset:
1. **Grayscale Conversion**: Converts RGB images to single-channel grayscale using luminance formulas.
2. **Gaussian Blur**: Applies a $3\times3$ kernel to smooth the image and reduce noise.
3. **Edge Detection**: Uses the Sobel operator (X and Y directions) to highlight object boundaries.
4. **Sharpening**: Applies a custom kernel to enhance edge contrast.
5. **Brightness Adjustment**: Converts the image to HSV color space, increases the Value (V) channel, and converts back to RGB.

Use the "Upload File" utility to upload `image_processor.py` to the GCP VM via SSH.

## 🚀 Usage & Execution
The `image_processor.py` script includes a fully automated benchmark suite. It will:
1. Verify the dataset exists.
2. Run a "Sanity Check" on a single image to ensure validity.
3. Execute the pipeline with 1, 2, 4, and 8 workers for both paradigms.
4. Generate a formatted report table.

Run the benchmark:
```
# install the cv2 requried package
pip install opencv-python-headless numpy

# execute the test
python3 image_processor.py --dir image-data
```

(Note: Replace `image-data` with your actual unzipped folder name if different).

## 📊 Performance Analysis
The script outputs a table similar to the one below. 
This data is used to calculate Speedup ($S = T_1 / T_N$) and Efficiency ($E = S / N$).
<br>Sample Output:

| Workers | MP Time(s) | MP Speedup | MP Eff | CF Time(s) | CF Speedup | CF Eff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 352.2515 | 1.00 | 1.00 | 352.0200 | 1.00 | 1.00 |
| **2** | 181.7335 | 1.94 | 0.97 | 178.2116 | 1.98 | 0.99 |
| **4** | 167.3402 | 2.11 | 0.53 | 169.2503 | 2.08 | 0.52 |
| **8** | 171.7337 | 2.05 | 0.26 | 165.0644 | 2.13 | 0.27 |  


---

## 👥 Team Members
* TAN JIA JOO (163573)
* TAN YI PEI (164767)
* TAN YIN XUAN (164467)
* TEOH KAI LUN (164277)




