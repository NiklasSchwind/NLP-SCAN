# Ready to use NLP-SCAN

This is a ready to use version of the NLP-SCAN unsupervised classification method developed in my master thesis "Learning to Classify Texts without Labels. A NLP Adaptation of SCAN". It is based on the code for the paper [DocSCAN: Unsupervised Text Classification via Learning from Neighbors](https://aclanthology.org/2022.konvens-1.4/), a previous approach to translating the [SCAN method](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_16) to NLP.  There can be code fragments from the original DocSCAN code still in this code. However, we added a new embedding method that is applicable to all classification domains and added the Fine-Tuning through Self-Labeling step from the original SCAN. 

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n NLPSCAN python=3.6
conda activate NLPSCAN

pip3 install -U sentence-transformers
conda install faiss-cpu -c pytorch
pip3 install -r requirements.txt
```

There is a shellscript called "setup_environment.sh" that does sets up the environment automatically. 

## Apply NLP-SCAN to unlabeled text dataset

1. Save unlabeled dataset as text file with one text per line 
2. Copy the path to the file into the file_path_data variable in the NLPSCAN script 
3. Write the path where NLP-SCAN should save the resulting classification of the dataset into the file_path_result variable in the NLPSCAN script
4. If needed, adjust the other options in the NLPSCAN or NLPSCAN_pipeline scripts
5. Create the NLPSCAN environment as described in the installation section
6. Activate the NLPSCAN environment with 
```shell
conda activate NLPSCAN
```
7. Start the NLPSCAN script with the following command
```shell
PYTHONPATH=src python NLPSCAN.py 
```
8. NLPSCAN will classify your unlabeled dataset and safe the classified version as a csv file with the columns "sentence" and "class"

