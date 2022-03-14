#source /home/sem/anaconda3/bin/activate
#conda activate trt



# vgg16



# python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 64 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_org  --compress_rate [0.]*13 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 128 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_org  --compress_rate [0.]*13 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 512 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_org  --compress_rate [0.]*13 --net vgg_16_bn



python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 64 --test_model_dir './pruned_models/vgg16/92.81/92.81.pth' --is_cpu --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 128 --test_model_dir './pruned_models/vgg16/92.81/92.81.pth' --is_cpu --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 512 --test_model_dir './pruned_models/vgg16/92.81/92.81.pth' --is_cpu --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn

python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 64 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_org --is_cpu --compress_rate [0.]*13 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 128 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_org --is_cpu --compress_rate [0.]*13 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir './data' --batch_size 512 --test_model_dir './checkpoints/vgg_16_bn.pt' --is_org --is_cpu --compress_rate [0.]*13 --net vgg_16_bn

# python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 64 --test_model_dir './pruned_models/vgg16/92.81/92.81.pth'  --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 128 --test_model_dir './pruned_models/vgg16/92.81/92.81.pth'  --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
# python test2.py --dataset 'cifar10' --data_dir ./data --batch_size 512 --test_model_dir './pruned_models/vgg16/92.81/92.81.pth'  --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn