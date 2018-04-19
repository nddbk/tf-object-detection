# tf-object-detection
Play with TensorFlow Object Detection


# Preparation

### Setup workspace

```
# create workspace folder
mkdir workspace
cd workspace

# create virtual environment
python3 -m venv cv-venv
source cv-venv/bin/activate
# install tensorflow or tensorflow-gpu
pip install tensorflow-gpu

# get source code
git clone https://github.com/ndaidong/tf-object-detection.git

cd tf-object-detection
pip install -r requirements.txt
./init.py

cd tflib
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
python tflib/object_detection/builders/model_builder_test.py

```

For more info:

- [TensorFlow Object Detection - Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- [Setup a lightweight environment for deep learning](https://medium.com/@ndaidong/setup-a-simple-environment-for-deep-learning-dc05c81c4914)


### Download dataset


```
# cd workspace
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
# cd workspace/tf-object-detection
./make_tfrecord.py -d ../vgg-faces-utils/output -e 100 -o temp/data

```

Parameters:

- `-d`, `--dir`: relative path to folder where we store labels and images
- `-e`, `--extract`: how many images we want to extract from the whole set. Default: `100`.
- `-o`, `--output`: relative path to folder where TFRecord files will be saved into. Default: `temp/data`
- `-l`, `--labelmap`: relative path to label map file. Default `configs/label_map.pbtxt`
- `-r`, `--ratio`: ratio of test set / training set. Default: `0.1` (1 image for test, 9 images for training)

For more info:

- [TensorFlow Object Detection - Preparing Inputs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)


### Get checkpoints

```
# cd workspace
wget http://49.156.52.21:7777/checkpoints/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -zxvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C tf-object-detection/temp/checkpoints
```


To find more pretrained models:

- [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)



# Training


```
# cd workspace/tf-object-detection
python tflib/object_detection/train.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco.config --train_dir=temp/models/ssd_mobilenet_v2/train
```

### Evaluation


```
# cd workspace/tf-object-detection
python tflib/object_detection/eval.py --logtostderr --pipeline_config_path=configs/ssd_mobilenet_v2_coco.config --checkpoint_dir=temp/models/ssd_mobilenet_v2/train --eval_dir=temp/models/ssd_mobilenet_v2/eval

```


### TensorBoard

```
# cd workspace/tf-object-detection
tensorboard --logdir=training:temp/models/ssd_mobilenet_v2/train,test:temp/models/ssd_mobilenet_v2/eval
```

For more info:

- [TensorFlow Object Detection - Running Locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)


# Export graph


```
# cd workspace/tf-object-detection
python tflib/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=configs/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix=temp/models/ssd_mobilenet_v2/train/model.ckpt-0 --output_directory=temp/output/ssd_mobilenet_v2_graph.pb
```


# Prediction


// coming soon


# License

The MIT License (MIT)

