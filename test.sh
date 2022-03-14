#source /home/sem/anaconda3/bin/activate
#conda activate trt

# vgg16
#python test_temp.py --data_dir ./data --batch_size 64 --compress_rate [0.]*99 --net vgg_16_bn
#python test_temp.py --data_dir ./data --batch_size 128 --compress_rate [0.]*99 --net vgg_16_bn
#python test_temp.py --data_dir ./data --batch_size 512 --compress_rate [0.]*99 --net vgg_16_bn
python test_temp.py --data_dir ./data --batch_size 64 --is_cpu --compress_rate [0.]*99 --net vgg_16_bn
python test_temp.py --data_dir ./data --batch_size 128 --is_cpu --compress_rate [0.]*99 --net vgg_16_bn
python test_temp.py --data_dir ./data --batch_size 512 --is_cpu --compress_rate [0.]*99 --net vgg_16_bn

#python test_temp.py --data_dir ./data --batch_size 64 --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
#python test_temp.py --data_dir ./data --batch_size 128 --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
#python test_temp.py --data_dir ./data --batch_size 512 --compress_rate [0.50]*7+[0.95]*5 --net vgg_16_bn
python test_temp.py --data_dir ./data --batch_size 64 --is_cpu --compress_rate [0.35]*7+[0.80]*5 --net vgg_16_bn
python test_temp.py --data_dir ./data --batch_size 128 --is_cpu --compress_rate [0.35]*7+[0.80]*5 --net vgg_16_bn
python test_temp.py --data_dir ./data --batch_size 512 --is_cpu --compress_rate [0.35]*7+[0.80]*5 --net vgg_16_bn


