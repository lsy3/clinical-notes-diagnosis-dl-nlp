echo "id,584,276,518,414,428,272,401,250,285,427,features" > data/DATA_TFIDF_HADM_TOP10CAT.csv
FIRST=1
for f in data/DATA_TFIDF_HADM_TOP10CAT/*.csv;
do
    if [ "$FIRST" = "1" ]; then
        echo "Processing $f file (first)";
	FIRST=0
        cat $f > data/DATA_TFIDF_HADM_TOP10CAT.csv
    else
        echo "Processing $f file";
        sed '1d' $f >> data/DATA_TFIDF_HADM_TOP10CAT.csv
    fi
done
