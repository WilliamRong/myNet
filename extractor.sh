#!/usr/bin/env bash
#extract features per RGB and Flow by Resnet
cd ./extracted_rgb_flow_cad_preprocessed4resnet/
dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for i in $dir
do
cd ./$i
sub_dir=$(ls -l ./ |awk '/^d/ {print $NF}')
for j in $sub_dir
do
cd ./$j
python /media/a5/image3/rw/models/myNet/extractor.py --data_dir ./RGB --features_dir /media/a5/image3/rw/models/myNet/features_resnet/$i/$j/RGB --modality RGB

python /media/a5/image3/rw/models/myNet/extractor.py --data_dir ./Flow --features_dir /media/a5/image3/rw/models/myNet/features_resnet/$i/$j/Flow --modality Flow
cd ..
done
cd ..
done
