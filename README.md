# cse6250-final-project

Proposal Report: [overleaf link](https://www.overleaf.com/8371794wnkynjkydwsn)

### Todo List
1. Data Preprocessing
    1. top 10 all
    1. top 50 all
    1. top 10 category
    1. top 50 category
1. Feature Selection
    1. bag of words
    1. word2vec
1. Model Training and Testing
    1. LR
    1. Random Forest
    1. Feed Forward NN
    1. RNN
    1. LSTM
    1. GRU

### Environment Setup
1. conda env create -f environment.yml
1. jupyter toree install --user --spark_home=<complete path>/spark-2.1.0-bin-hadoop2.7 --interpreters=PySpark
    1. If you experience error in running the PySpark kernel. Make sure you put "complete path", not relative path.
    1. If toree is not yet installed (should be included in environment.yml), run the ff. command:
        1. pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
        1. [github link](https://github.com/apache/incubator-toree)
1. data cleaning
    1. gawk -v RS='"' 'NR % 2 == 0 { gsub(/\n/, "|") } { printf("%s%s", $0, RT) }' NOTEEVENTS.csv > NOTEEVENTS2.csv
        1. this approach did not completely work for me -Luke

### Folder Structure
* code: all source code
    * data (you should download and extract the following here)
        * DIAGNOSES_ICD.csv
        * NOTEEVENTS.csv
        * PATIENTS.csv (will probably not use this)
* literature: important papers (pretty much the same w/ what is on slack)
* proposal: other proposal related documents
