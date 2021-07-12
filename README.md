# HET-GMP: a Graph-based System Approach to Scaling Large Embedding Model Training (SIGMOD 2022)

## Installation
1. Clone this respository.

2. prepare build requirements:

```shell
# make sure cuda (>=10.1) is already installed in /usr/local/cuda
# create a new conda environment
conda install -c conda-forge \
cmake=3.18 zeromq=4.3.2 pybind11=2.6.0 thrust=1.11 cub=1.11 nccl=2.9.9.1 cudnn=7.6.5 openmpi=4.0.3
```

3. build

```shell
mkdir build && cd build && cmake .. && make -j && cd ..
source hetu.exp # this edits PYTHONPATH
```

4. Some python packages and necessary to run the datasets processing and training script below.

```shell
pip install --upgrade-strategy only-if-needed \
scipy sklearn numpy pyyaml argparse pandas tqdm
```

## Download and process datasets

Download criteo datasets from https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310

```shell
# in repo root directory
mkdir -p ~/hetuctr_dataset/criteo
# put your downloaded dac.tar.gz in ~/hetuctr_dataset/criteo

# copy data process script
cp ./examples/models/load_data.py ~/hetuctr_dataset/criteo
# copy graph partition script
cp ./hetuCTR/experimental/partition.py ~/hetuctr_dataset/criteo

# --------------------------------------------------------

cd ~/hetuctr_dataset/criteo

# process criteo data
python3 load_data.py

# run graph partition
# Note : you can skip this step if you only use one gpu or want to use random partition
python3 partition.py -n 8 -o criteo_partition_8.npz --rerun 5
```

Finally, you can find 6 npy file which are processed train data and a npz file which is the partition reuslt.

## Train models

Run this script to train on a single GPU:

```shell
python3 examples/hetuctr.py \
--dataset criteo --model wdl \
--batch_size 8192 --iter 1000000 --embed_dim 16 \
--val --eval_every 10000
```

Train on 8 GPUs with partition and staleness

```shell
# in repo root directory
mpirun --allow-run-as-root -np 8 \
python3 examples/hetuctr.py \
--dataset criteo --model wdl \
--batch_size 8192 --iter 1000000 --embed_dim 128 \
--partition ~/hetuctr_dataset/partition/criteo_partition_8.npz \
--store_rate 0.01 --bound 100 \
--val --eval_every 10000
```

Arguments :

​	--embed_dim : the dimension for each embedding index

​    --partition : assign a partition file, if no partition is provided, random partition is used

​    --store_rate : the amount of mirror embeddings , 0.01 means selects top 1% priority embedding as mirror embeddings on each worker

​    --bound : the staleness bound, set to 0 for BSP training, use values 10, 100 for better performance.

​    --val, --eval_every : whether to perform evaluation

​    --iter : how many iterations to run

​    --batch_size : batch size on each worker

​    --model : wdl for WideDeep model, dcn for Deep&Cross model

