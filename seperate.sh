#!/usr/bin/env bash
#split rgb and flow into two folders respectively
dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for i in $dir
do
cd ./$i
sub_dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for j in $sub_dir
do
cd ./$j
mkdir RGB
mkdir Flow
mv img* ./RGB
mv flow* ./Flow
cd ..
done
cd ..
done
