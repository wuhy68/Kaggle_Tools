./distributed_train.sh 4 ../datasets --model efficientnet_b4 --pretrained --sched cosine --epoch 20 --batch-size 8 -j 4 --num-classes 5 --remode pixel --drop 0.4 --drop-path 0.2 --img-size 512 --opt sgdp --momentum 0.95 --weight-decay 0.0001