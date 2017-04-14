#array=( data/DATA_HADM data/DATA_HADM_CLEANED )
array=( data/DATA_HADM_CLEANED )
#array=( data/DATA_HADM_TOP50 ) 
#array=( data/DATA_TFIDFV0_HADM_TOP10_train data/DATA_TFIDFV0_HADM_TOP10_val data/DATA_TFIDFV0_HADM_TOP10_test data/DATA_TFIDFV0_HADM_TOP10CAT_train data/DATA_TFIDFV0_HADM_TOP10CAT_val data/DATA_TFIDFV0_HADM_TOP10CAT_test )
#array=( data/DATA_TFIDFV0_HADM_TOP50_train data/DATA_TFIDFV0_HADM_TOP50_val data/DATA_TFIDFV0_HADM_TOP50_test ) 
#array=( data/DATA_HADM_TOP10 data/DATA_HADM_TOP10CAT data/DATA_TFIDFV0_HADM_TOP10_train data/DATA_TFIDFV0_HADM_TOP10_val data/DATA_TFIDFV0_HADM_TOP10_test )

for TARGET_FILE in "${array[@]}"
do
	FIRST=1
	for f in $TARGET_FILE/*.csv;
	do
	    if [ "$FIRST" = "1" ]; then
		echo "Processing $f file (first)";
		FIRST=0
		cat $f > $TARGET_FILE.csv
	    else
		echo "Processing $f file";
		sed '1d' $f >> $TARGET_FILE.csv
	    fi
	done
done
