array=( data/DATA_HADM data/DATA_HADM_CLEANED )

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
