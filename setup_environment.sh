conda create -n NLPSCAN python=3.6
conda activate NLPSCAN

pip install -U sentence-transformers
conda install faiss-cpu -c pytorch

pip install -r requirements.txt
