# NLP-SCAN

NLP-SCAN is an unsupervised classification method using an innovative embedding technique that can be tailored to various classification tasks. It is based on the code for the paper [DocSCAN: Unsupervised Text Classification via Learning from Neighbors](https://aclanthology.org/2022.konvens-1.4/), which translates the [SCAN method](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_16) to NLP. However, NLP-SCAN additionally incorporates the Fine-Tuning through Self-Labeling step from the original SCAN method, resulting in increased accuracy.

## Advantages
NLP-SCAN is a transformer-embedding-based unsupervised text classification algorithm with the following main advantages:
- Trains a text classification model on unlabeled text datasets
- The semantic focus of the classification model can be tailored to the task using the Indicative Sentence embedding method
- Fast, accurate, and requires minimal human input

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and Linux, the environment can be installed with the following commands:
```shell
conda create -n nlpscan python=3.8 
conda activate nlpscan
pip install -r requirements.txt
```

## Tutorial: Applying NLP-SCAN to Unlabeled Text Data

This tutorial will guide you through the steps to apply NLP-SCAN to your unlabeled text dataset.

### Step 1: Prepare Your Dataset
Ensure your text data is saved as a text file with one text per line.

### Step 2: Configure File Paths
Open the `NLPSCAN.py` script. Set the path to your dataset in the `file_path_data` variable and the path for the result in the `file_path_result` variable.

### Step 3: Adjust Configurations
If needed, adjust other configurations within the `NLPSCAN.py` script to suit your specific requirements.

### Step 4: Create the NLP-SCAN Environment
Follow the installation instructions to create the NLP-SCAN environment using Anaconda.

### Step 5: Activate the Environment
Activate the NLP-SCAN environment:
```shell
conda activate nlpscan
```
### Step 6: Run the NLP-SCAN Script
Execute the NLP-SCAN script:
```shell
python NLPSCAN.py
```
### Step 7: Retrieve Your Results
NLP-SCAN will process your dataset and save the classified results as a CSV file with "sentence" and "class" columns.

