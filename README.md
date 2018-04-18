# tf-ssd-mobilenet
Train Face Detection model with TensorFlow Object Detection + SSD + MobileNet


# Preparation

### Setup workspace

```
# create workspace folder
mkdir face-detection
cd face-detection

# create virtual environment
python3 -m venv venv
source venv/bin/activate

# get source code
git clone https://github.com/ndaidong/tf-ssd-mobilenet.git

cd tf-ssd-mobilenet
pip install -r requirements.txt
./init.py

```

For more info:

- [TensorFlow Object Detection - Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)



### Download dataset


```
# cd face-detection
wget http://49.156.52.21:7777/dataset/vgg_face_dataset.tar.gz
git clone https://github.com/ndaidong/vgg-faces-utils.git
tar -zxvf vgg_face_dataset.tar.gz -C vgg-faces-utils
pip install -r vgg-faces-utils/requirements.txt
cd vgg-faces-utils
./script.py -d vgg_face_dataset/files -o output
```

For more info:

- [Download and annotate images from VGG Faces dataset](https://github.com/ndaidong/vgg-faces-utils#usage).


### Generate TFRecord files

```
# cd face-detection/tf-ssd-mobilenet
./make_tfrecord.py -d ../vgg-faces-utils/output -e 100 -o temp/data -r 0.1

```

Parameters:

- `-d`, `--dir`: relative path to folder where we store labels and images
- `-e`, `--extract`: how many images we want to extract from the whole set. Default: `100`.
- `-o`, `--output`: relative path to folder where TFRecord files will be saved into. Default: `temp/data`
- `-r`, `--ratio`: ratio of test set / training set. Default: `0.1` (1 image for test, 9 images for training)

For more info:

- [TensorFlow Object Detection - Preparing Inputs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)


### Get checkpoints

```
# cd face-detection
wget http://49.156.52.21:7777/checkpoints/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -zxvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C temp/checkpoints
```

To find more pretrained models:

- [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)


# Training


```
# cd face-detection

# setup tensorflow models
git clone https://github.com/tensorflow/models.git
cd ../models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# back to tf-ssd-mobilenet
cd ../../tf-ssd-mobilenet

# check if Object Detection works
python ../models/research/object_detection/builders/model_builder_test.py

# train
python ../models/research/object_detection/train.py --logtostderr --pipeline_config_path=configs/training_pipeline.config --train_dir=temp/models/v1/train
```

### Evaluation


```
# cd face-detection/tf-ssd-mobilenet
python ../models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/training_pipeline.config --checkpoint_dir=temp/models/v1/train --eval_dir=temp/models/v1/eval

```


### TensorBoard

```
tensorboard --logdir=training:temp/models/v1/train,test:temp/models/v1/eval
```

For more info:

- [TensorFlow Object Detection - Running Locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)


# Prediction


// coming soon


# License

The MIT License (MIT)

