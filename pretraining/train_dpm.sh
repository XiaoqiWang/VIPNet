python ./multi_gpu_train.py -a botnet  -p 1000 \
      --pretrain_path '../../kadis6000k' \
      --kadid_test_path '../../iqadataset/kadid10k' \
      --epochs 40 -b 320 --lr 0.05 --workers 16 \
      --dist-url 'tcp://127.0.0.1:1521' \
      --dist-backend 'nccl' --multiprocessing-distributed \
      --world-size 1 --rank 0
