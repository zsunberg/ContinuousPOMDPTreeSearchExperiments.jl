#!/bin/bash

num=$1
shift

#Day_Date_Month_hour_minute
now=`date +"%a_%d_%b_%H_%M"`
datadir="data"
fname="$datadir/cpp_run_$now.txt"
echo "Will eventually write to $fname..."
echo "using args $@"

mkdir "/tmp/cpp_run_$now"

# for i in $(seq 1 $num); do
#     /home/zsunberg/Devel/despot/examples/cpp_models/tag/lasertag -r $RANDOM $@ \
#         | gzip > "/tmp/cpp_run_$now/sim_$i.gz" && echo "Completed $i" &
# done


parallel --progress /home/zsunberg/Devel/despot/examples/cpp_models/tag/lasertag -r {1} $@ \
    '|' gzip '>' /tmp/cpp_run_$now/sim_{1}.gz ::: $(seq 1 $num)

echo "Finished simulations. Combining..."

for i in $(seq 1 $num); do
    gunzip -c "/tmp/cpp_run_$now/sim_$i.gz" >> $fname;
done

echo "Getting rewards..."

grep "Total discounted" $fname | awk '{print $5}' > "$datadir/cpp_run_${now}_rewards.txt"

grep "Total discounted" $fname |  awk '{ s += $5; s2 += $5^2; } END { printf "Average total discounted reward (stderr) = %f (%f)\n", s/NR, sqrt(s2 / NR^2 - s^2 / NR^3) }'

echo "Gzipping Log..."

gzip $fname
