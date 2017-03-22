echo "id,4019,2724,25000,4280,41401,53081,51881,42731,5849,5990,features" > data/DATA_TFIDF_HADM_TOP10.csv
FIRST=1
for f in data/DATA_TFIDF_HADM_TOP10/*.csv;
do
    if [ "$FIRST" = "1" ]; then
        echo "Processing $f file (first)";
	FIRST=0
        cat $f > data/DATA_TFIDF_HADM_TOP10.csv
    else
        echo "Processing $f file";
        sed '1d' $f >> data/DATA_TFIDF_HADM_TOP10.csv
    fi
done
