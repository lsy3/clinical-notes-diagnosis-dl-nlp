# CSE6250 Final Project: Learning to Diagnose using Clinical Notes

Members:
* Jinmiao Huang
* Cesar Osorio
* Luke Wicent Sy

To be deleted soon
1. Final Report (draft): [overleaf link](https://www.overleaf.com/8371794wnkynjkydwsn#/31606347/)
1. Cesar's google drive with word2vec and doc2vec models
    1. (https://drive.google.com/open?id=0B5wTZcZsz2x7eVhaNkJoNkNWaWs)

### Environment Setup (local)
1. conda env create -f environment.yml
1. Install spark or download spark binary from [here](https://spark.apache.org/downloads.html)
1. pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
    * the above command should install toree. if it fails, refer to [github link](https://github.com/apache/incubator-toree).
    * note that toree was not included in environment.yml because including it there didn't work for me before. 
1. jupyter toree install --user --spark_home=<complete path>/spark-2.1.0-bin-hadoop2.7 --interpreters=PySpark
1. Extract the ff. files to the directory "code/data":
    * DIAGNOSES_ICD.csv (from bd4h piazza)
    * NOTEEVENTS-2.csv (cleaned version of NOTEEVENTS.csv; can be downloaded [here](https://drive.google.com/open?id=0B7IQxoKP3KPGWmFiUGlNTTBuWXM)
    * D_ICD_DIAGNOSES.csv (from bd4h piazza)
1. To run data preprocessing, data statistics, and ipynb related stuff, start the jupyter notebook. Don't forget to set the kernel to "Toree Pyspark".
    * jupyter notebook
1. To run the deep learning experiments, follow the corresponding guide below.

### Environment Setup (azure)
1. Setup Docker w/ GPU following [this guide](https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Azure)
1. Using Azure's portal, select the vm's firewall (in my case, it showed "azure01-firewall" in "all resources"), then "allow" port 22 (ssh) and 8888 (jupyter) for both inbound and outbound.
1. You can ssh the VM through one of the ff:
    * docker-machine ssh azure01
    * ssh docker-user@public_ip_addr
1. Spark can be installed by following the instructions in "Environment Setup (local), but note that this will not be as powerful as HDInsights. I recommend taking advantage of the VM's large memory by setting the spark memory to a higher value (<spark folder>/conf/spark-defaults.conf)
1. If you have a jupyter notebook running in this VM, you can access via http://public_ip_addr:8888/
1. To enable the GPUs for deep learning, follow the instructions in the tensorflow website [link](https://www.tensorflow.org/install/install_linux)
    * you can check the GPUs' status by "nvidia-smi"


###Jupyter (on Docker VM)
To be deleted soon
1. Go to http://52.179.1.29:8888/
1. If asked for a password, enter 'jupyter'.
1. Logging in via SSH (see Docker w/ GPU).
    * If you can't connect to jupyter, the VM might be down or it might need to be initialized via 'screen jupyter notebook'
1. Uploading a file:
    * scp -i path-to-git-root/id_rsa file-to-upload docker-user@52.179.1.29:~

### Docker w/ GPU (for deep learning) [azure setup guide]()
To be deleted soon
1. make sure you have id_rsa (update your git repo using 'git pull')
1. ssh -i path-to-git-root/id_rsa docker-user@52.179.1.29
1. source activate cse6250

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
* Prerequirest: Keras + Tensorflow, or Keras + Theano
* all the models are specified in `dl_models.py` file. 
* run `tfidf_preprocessing` to prapare the data for deep learning training and testing use. 
* training:
    * Run training script with input arguments: `python tfidf_train.py --epoch 10 --batch_size 128 --model_name nn_model_1 --pre_train False`
    * `--epoch`: number of passes over the entire dataset, default: `epoch = 50`
    * `--batch_size`: number of samples in batch training, default: `batch_size = 128`
    * `--model_name`: provide function name in `dl_models.py` to call the corresonding model. We also save the trained weights use the provided model name, as `weights_{model_name}.h5`, default: `model_name = 'nn_model_1'`
    * `--pre_train`: by default is False, if set to True, model will first load the pretrained weights from `weights_{model_name}.h5`, then continue the training. default: `pre_train = False`
    *  You can also run training with default arguments: `pythno tfidf_train.py`, 
    *  NOTE: TensorFlow will attempt to use (an equal fraction of the memory of) all GPU devices that are visible to it. If you want to run different sessions on different GPUs, you should do the following.
        *   Run each session in a different Python process.
        *   Start each process with a different value for the CUDA_VISIBLE_DEVICES environment variable. For example, if you want to test 2 different models specified in `dl_models.py` for tfidf data and you have 4 GPUs, you could run the following:
        *   `$ CUDA_VISIBLE_DEVICES=0 python tfidf_train.py --epoch 10 --batch_size 128 --model_name nn_model_1 --pretrain False` # Uses GPU 0.
        *   `$ CUDA_VISIBLE_DEVICES=1 python tfidf_train.py --epoch 10 --batch_size 128 --model_name nn_model_1 --pretrain False` # Uses GPU 1.
        *   You can also use multiple gpus in one script. (not supported in our code)
        *   `$ CUDA_VISIBLE_DEVICES=2,3 python some_script.py`  # Uses GPUs 2 and 3.
        *   The GPU devices in TensorFlow will still be numbered from zero (i.e. "/gpu:0" etc.), but they will correspond to the devices that you have made visible with CUDA_VISIBLE_DEVICES

* testing:
    * Test model with specified model name: `$ CUDA_VISIBLE_DEVICES=0 python tfidf_test.py --model_name nn_model_1 --batch_size 128`   
    * `--model_name` default: `model_name = 'nn_model_1'`
    * `--batch_size` default: `batch_size = 128`

### Todo List
1. Data Preprocessing
    1. ~~top 10 all~~ 
    1. top 50 all
    1. ~~top 10 category~~
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
