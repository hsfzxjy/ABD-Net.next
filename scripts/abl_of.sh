export no_save=1

for i in {1..9}
do
    GPU=0 of_beta="$i"e-5 bash scripts/train_of.sh debug &
    GPU=1 of_beta="$i"e-6 bash scripts/train_of.sh debug &
    GPU=2 of_beta="$i"e-7 bash scripts/train_of.sh debug &
    GPU=3 of_beta="$i"e-8 bash scripts/train_of.sh debug &
    wait
done

GPU=3 of_beta=0 bash scripts/train_of.sh debug &
wait