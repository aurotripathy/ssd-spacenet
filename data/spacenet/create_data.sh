cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
echo "+++Current dir" $cur_dir
root_dir=$cur_dir/../..
echo "+++Root dir" $root_dir

cd $root_dir

redo=1
data_root_dir="$HOME/spacenet-data"
echo "+++data_root_dir" $data_root_dir
dataset_name="spacenet"
echo "+++dataset_name" $dataset_name
mapfile="$root_dir/data/$dataset_name/labelmap_spacenet.prototxt"
echo "+++mapfile" $mapfile

anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=tif --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  echo "Working on subset" $subset
  echo "Where is subset" $root_dir/data/$dataset_name/$subset.txt
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile \
         --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height \
         --check-label $extra_cmd $data_root_dir \
         $root_dir/data/$dataset_name/$subset.txt \
         $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
