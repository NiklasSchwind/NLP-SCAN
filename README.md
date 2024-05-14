# NLP-SCAN

NLP-SCAN is an unsupervised classification method using an innovative embedding method that can tailor the NLPSCAN to all kinds of classification task. It is based on the code for the paper [DocSCAN: Unsupervised Text Classification via Learning from Neighbors](https://aclanthology.org/2022.konvens-1.4/), a previous approach to translating the [SCAN method](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_16) to NLP. However, NLPSCAN additionally translated the Fine-Tuning through Self-Labeling step from the original SCAN to NLP resulting in increase accuracy.  

## Advantages
NLP-SCAN is a transformer-embedding-based unsupervised text classification algorithm with the following main advantages:
- It trains a text classification model on unlabeled text datasets 
- The semantic focus of the classification model can be tailored to the classification task by using the Indicative Sentence embedding method
- It is fast, accurate and needs minimal input by a human

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n nlpscan python=3.8 
conda activate nlpscan
pip install -r requirements.txt
```

## Apply NLP-SCAN to unlabeled text dataset

1. Save unlabeled text dataset as text file with one text per line 
2. Copy the path to the file into the file_path_data variable in the NLPSCAN.py script 
3. Write the path where NLP-SCAN should save the resulting classification into the file_path_result variable in the NLPSCAN script
4. If needed, adjust the other configurations in the NLPSCAN script 
5. Create the NLPSCAN environment using the installation guide
6. Activate the NLPSCAN environment with 
```shell
conda activate nlpscan
```
7. Start the NLPSCAN script with the following command
```shell
python NLPSCAN.py 
```
8. NLPSCAN will classify your unlabeled dataset and safe the classified version as a csv file with the columns "sentence" and "class" 

