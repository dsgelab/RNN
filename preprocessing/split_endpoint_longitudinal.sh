#!/bin/bash
###############################################################################
#
# A simple script for spliting endpoint longitudinal file (sorted by ID) to 
# chunks containing 300000 IDs each for the FinnGen project.
# 
# A file of 7166416 IDs is splt into 24 files
# 
# Author: Andrius Vabalas (andrius.vabalas@helsinki.fi)
#
###############################################################################
set -x

ENDPOINT_LONGITUDINAL="/data/processed_data/endpointer/longitudinal_endpoints_2021_12_20_no_OMITS.txt"

loop=1
line=1
for count in {1..24..1} #for count in {0000001..7166416..1000}
do
        echo "$loop"
        a=$((count*300000+1))
        if [ $count -gt 3 ]; then b="FR"; else b="FR0"; fi
        if [ $count -eq 4 ]; then a=$((count*300000+2)); fi
        if [ $count -eq 8 ]; then a=$((count*300000+2)); fi
        if [ $count -eq 11 ]; then a=$((count*300000+2)); fi
        if [ $count -eq 19 ]; then a=$((count*300000+2)); fi
        ID="${b}${a}"
        echo "$ID"
        if [ $count -eq 24 ]
        then
                patt=$line",$ p"
        else
                z=$(grep -n -o -m 1 $ID $ENDPOINT_LONGITUDINAL | cut -f1 -d:)
                line1=$(($z-1))
                patt=$line","$line1"p;"$line1"q"
        fi
        fname="endpoint_longitudinal.txt."$loop
        if [ $count -eq 1 ]
        then
                sed -n -e "$patt" "$ENDPOINT_LONGITUDINAL" > "$fname"
        else
                sed -n -e '1p' -e "$patt" "$ENDPOINT_LONGITUDINAL" > "$fname"
        fi
        line=$z
        loop=$(($loop+1))
done

