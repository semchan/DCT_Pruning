#source /home/sem/anaconda3/bin/activate
#conda activate trt

# vgg16


python3 test.py --dataset 'cifar10' --data_dir './data' --batch_size 64 --test_model_dir './pruned_models/vgg16/[0.50]*7+[0.95]*5/92.81.pth' --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn



