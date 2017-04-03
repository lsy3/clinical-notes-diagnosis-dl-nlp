# cse6250-final-project

Proposal Report: [overleaf link](https://www.overleaf.com/8371794wnkynjkydwsn#/31606257/)
Final Report (draft): [overleaf link](https://www.overleaf.com/8371794wnkynjkydwsn#/31606347/)

### Environment Setup
1. conda env create -f environment.yml
1. jupyter toree install --user --spark_home=<complete path>/spark-2.1.0-bin-hadoop2.7 --interpreters=PySpark
    1. If you experience error in running the PySpark kernel. Make sure you put "complete path", not relative path.
    1. If toree is not yet installed (should be included in environment.yml), run the ff. command:
        1. pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
        1. [github link](https://github.com/apache/incubator-toree)
1. extract the ff. data to code/data:
    1. DIAGNOSES_ICD.csv
    1. NOTEEVENTS.csv - a cleaned and sample version can be downloaded [here](https://drive.google.com/open?id=0B7IQxoKP3KPGWmFiUGlNTTBuWXM)
    1. DATA_TFIDF_HADM_TOP10.csv - tfidf data based on hadm_id<->top10 icd9codes [download link](https://drive.google.com/open?id=0B7IQxoKP3KPGWmFiUGlNTTBuWXM)
1. jupyter notebook
    1. Console command to run the notebook. Don't forget to set the kernel to "Toree Pyspark".

### HDInsight (for pyspark related stuff)
1. Logging in to Jupyter
    * click this [link](https://cse6250-fp.azurehdinsight.net/jupyter/tree) using your web browser
    * Username: admin
    * Password: cse6250FP!
1. Logging in via SSH (which you'll most likely not need)
    * ssh sshuser@cse6250-fp-ssh.azurehdinsight.net
    * Password: cse6250FP!
1. Uploading a file. Based on [link](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-upload-data#commandline) 
    * I tried Azure Command-Line Interface and Azure Storage Explorer, but for some reason, I can't upload using those methods. I just have this "uploading bar" with upload speed of 0 kbps.
    * I used the hadoop command line by typing the following on console
        * scp file-to-upload sshuser@cse6250-fp-ssh.azurehdinsight.net:~
        * ssh sshuser@cse6250-fp-ssh.azurehdinsight.net
        * hadoop fs -copyFromLocal data.txt /example/data/data.txt

### Docker w/ GPU (for deep learning) [azure setup guide](https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Azure)
1. make sure you have id_rsa (update your git repo using 'git pull')
1. ssh docker-user@40.71.188.95 -i path-to-git-root/id_rsa
1. source activate cse6250
    * to enter conda work environment.. although I haven't installed keras yet.. only tensorflow

### Folder Structure
* code: all source code
    * data (you should download and extract the following here)
        * DIAGNOSES_ICD.csv
        * NOTEEVENTS.csv
        * PATIENTS.csv (will probably not use this)
* literature: important papers (pretty much the same w/ what is on slack)
* proposal: other proposal related documents
* 

### Training and Testing for Keras Deep Learning Models
* all the models are written in `dl_models.py` file. 
* training:
    * `python tfidf_train.py --epoch 10 --batch_size 128 --model_name nn_model_1 --gpu 0 --pre_train False`
    * --epoch: number of passes over the entire dataset
    * --batch_size: number of samples in batch training
    * --model_name: provide function name in `dl_models.py` to call the corresonding model, the trained model weights will also be saved as `weights_{model_name}.h5`
    * --gpu: specify which gpu to use
    * --pre_train: by default is False, if set to True, model will first load the pretrained weights from `weights_{model_name}.h5`, then continue the training. 
* testing:
    * `python tfidf_test.py --batch_size 128 --model_name nn_model_1 --gpu 0`   
    * 
    * 

### Todo List
1. Data Preprocessing
    1. ~~top 10 all~~ (run tfidf_preprocessing to prapare the data for deep learning training and testing use) 
    1. top 50 all (code ready)
    1. top 10 category
    1. top 50 category
    1. consider doing advance test and train split
        * maintain the test-train ratio of each category instead of just the test-train ratio of the whole dataset
1. Feature Selection
    1. bag of words
        * consider removing "very frequent words / stop words of the medical field" (Cesar's suggestion) 
    1. word2vec
1. Model Training and Testing
    1. Logistic Regression
        * implemented using spark.ml instead of spark.mllib. [reason in this link](http://stackoverflow.com/questions/30231840/difference-between-org-apache-spark-ml-classification-and-org-apache-spark-mllib)
    1. Random Forest
    1. Feed Forward NN
    1. RNN
    1. LSTM
    1. GRU
