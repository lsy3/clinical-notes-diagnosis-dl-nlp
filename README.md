# An Empirical Evaluation of CNNs and RNNs for ICD-9 Code Assignment using MIMIC-III Clinical Notes

* Members: Jinmiao Huang, Cesar Osorio, and Luke Wicent Sy (all three provided equal contribution)
* Publication: [arxiv (2018)](https://arxiv.org/abs/1802.02311), [elsevier (updated 2019)](https://doi.org/10.1016/j.cmpb.2019.05.024)
* If you used this code in your work, please cite the following publication:
      Huang, J., Osorio, C., & Sy, L. W. (2019). An empirical evaluation of deep learning for ICD-9 code assignment using MIMIC-III clinical notes. Computer Methods and Programs in Biomedicine, 177, 141–153. https://doi.org/10.1016/J.CMPB.2019.05.024

### General Pipeline
1. (optional) cleaned NOTEEVENTS.csv using postgresql. imported NOTEEVENTS.csv by modifying [mimic iii github](https://github.com/MIT-LCP/mimic-code) and using the commands "select regexp_replace(field, E'[\\n\\r]+', ' ', 'g' )". the cleaned version (NOTEEVENTS-2.csv) can be downloaded in the google drive mentioned in "Environment Setup (local)"
1. run preprocess.ipynb to produce DATA_HADM and DATA_HADM_CLEANED.
1. run describe_icd9code.ipynb and describe_icd9category.ipynb to produce the descriptive statistics.
1. (optional) run word2vec-generator.ipynb to produce the word2vec models
1. run feature_extraction_seq.ipynb and feature_extraction_nonseq.ipynb to produce the input features for the machine learning and deep learning classifiers.
1. run ml_baseline.py to get the results for Logistic Regression and Random Forest.
1. run nn_baseline_train.py and nn_baseline_test.py to get the results for Feed-Forward Neural Network.
1. run wordseq_train.py and wordseq_test.py to get the results for Conv1D, RNN, LSTM and GRU (refer to 'help' or the guide below on training and testing for Keras Deep Learning Models)

### Training and Testing for Feed Forward Neural Network
* Prerequirest: Keras + Tensorflow, or Keras + Theano
* models are specified in `nn_baseline_models.py`
* run `nn_baseline_preprocessing` to prapare the data for training and testing use. 
* Training:
    * You can also run training with default arguments: `pythno nn_baseline_train.py`,
    * Or run training script with customized input arguments: `python nn_baseline_train.py --epoch 10 --batch_size 128 --model_name nn_model_1 --pre_train False`
    * Please refer to `parse_args()` function in `nn_baseline_train.py` for the full list of the input arguments 
    
* Testing:
    * Test model with default model and data file: `python tfidf_test.py`   
    * Please refer to `parse_args()` function in `nn_baseline_train.py` for the full list of the input arguments

### Training and Testing for Recurrent and Convolution Neural Network
* Similar to Feed Forward Neural Network, users can run the training and tesing with the default settings in `wordseq_train.py` and `wordseq_test.py`. All the model architectures are specified in `wordseq_models.py`

### Environment Setup (local)
1. conda env create -f environment.yml
1. Install spark or download spark binary from [here](https://spark.apache.org/downloads.html)
1. pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
    * the above command should install toree. if it fails, refer to [github link](https://github.com/apache/incubator-toree).
    * note that toree was not included in environment.yml because including it there didn't work for me before. 
1. jupyter toree install --user --spark_home=<complete path>/spark-2.1.0-bin-hadoop2.7 --interpreters=PySpark
1. Extract the ff. files to the directory "code/data":
    * DIAGNOSES_ICD.csv (from MIMIC-III database)
    * NOTEEVENTS-2.csv (cleaned version of MIMIC-III NOTEEVENTS.csv, replaced '\n' with ' ') [link](https://drive.google.com/file/d/0B7IQxoKP3KPGU25pY1hNazZPYkE/view?usp=sharing&resourcekey=0-rCepn0-MTOP2_cj-A2laew)
    * D_ICD_DIAGNOSES.csv (from MIMIC-III database)
    * model_word2vec_v2_*dim.txt (generated word2vec)
    * bio_nlp_vec/PubMed-shuffle-win-*.txt [Download here](https://github.com/cambridgeltl/BioNLP-2016) (you will need to convert the .bin files to .txt. I used gensim to do this)
    * model_doc2vec_v2_*dim_final.csv (generated word2vec)
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
