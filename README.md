[//]: <links>
[smmd]: https://arxiv.org/abs/1805.11565

# Scaled MMD GANs


Official Tensorflow implementation for reproducing results of [On gradient regularizers for MMD GANs][smmd].


## Setup
### Install:

`pip install -r requirements.txt`

The GPU compatible version of tensorflow is required for this code to work.


### Download CelebA dataset:

```
cd scripts
OUTPUT_DATA_DIR=/path/to/output/directory/
python scripts/download.py celebA -- $OUTPUT_DATA_DIR
```

### Download ImageNet dataset:
Please download ILSVRC2012 dataset from http://image-net.org/download-images

### Preprocess ImageNet dataset:
```
IMAGENET_TRAIN_DIR=/path/to/imagenet/train/ 
PREPROCESSED_DATA_DIR=/path/to/save_dir/
TFRECORDS_DATA_DIR=/path/to/output/tfrecords 
bash preprocess.sh $IMAGENET_TRAIN_DIR $PREPROCESSED_DATA_DIR
build_imagenet_data --train_directory=$PREPROCESSED_DATA_DIR --output_directory=$TFRECORDS_DATA_DIR
```

### Download inception model: 

`python source/inception/download.py --outfile=datasets/inception_model`


## Training


### Unsupervised image generation of 64x64 ImageNet:
```
DATADIR=/path/to/datadir/
OUTDIR=/path/to/outputdir/
CONFIG=configs/imagenet_smmd.yml
# multi-GPU: 3 GPUs
CUDA_VISIBLE_DEVICES=0,1,2 python gan/main.py -dataset imagenet -data_dir $DATADIR -name  -config_file $CONFIG -out_dir $OUTDIR -multi_gpu true
```


<p align="center">
	<img src="https://github.com/MichaelArbel/Scaled-MMD-GAN/blob/master/samples/imagenet.jpg">
</p>


For any question, please feel free to contact Michael Arbel (`michael.n.arbel@gmail.com`)

### References
Michael Arbel, Dougal J. Sutherland, Mikołaj Bińkowski, Arthur Gretton. *On gradient regularizers for MMD GANs*. [arXiv][smmd]

