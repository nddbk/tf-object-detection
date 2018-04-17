# tf-ssd-mobilenet
Train Face Detection model with TensorFlow Object Detection + SSD + MobileNet


# Usage

```
# create workspace folder
mkdir face-detection
cd face-detection

# create virtual environment
python3 -m venv venv
source venv/bin/activate

# get source code
git clone git@github.com:ndaidong/tf-ssd-mobilenet.git

pip install -r tf-ssd-mobilenet/requirements.txt
./init.py

```

# Preparation

### Download dataset


Please follow [the instruction here](https://github.com/ndaidong/vgg-faces-utils#usage).


```
# cd face-detection
wget http://49.156.52.21:7777/dataset/vgg_face_dataset.tar.gz
git clone https://github.com/ndaidong/vgg-faces-utils.git
tar -zxvf vgg_face_dataset.tar.gz -C vgg-faces-utils
pip install -r vgg-faces-utils/requirements.txt
cd vgg-faces-utils
./script.py -d vgg_face_dataset/files -o output
```


### Generate TFRecord files

```
# cd face-detection/tf-ssd-mobilenet
./make_tfrecord.py -d ../vgg-faces-utils/output -o temp/data -r 0.1

```

### Get checkpoints

```
# cd face-detection
wget http://49.156.52.21:7777/checkpoints/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -zxvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C temp/checkpoints
```


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

# Evaluation


```
# cd face-detection/tf-ssd-mobilenet
python ../models/research/object_detection/eval.py --logtostderr --pipeline_config_path=configs/training_pipeline.config --checkpoint_dir=temp/models/v1/train --eval_dir=temp/models/v1/eval

```


# TensorBoard

```
tensorboard --logdir=training:temp/models/v1/train,test:temp/models/v1/eval
```

# Prediction


// coming soon


# License

The MIT License (MIT)

