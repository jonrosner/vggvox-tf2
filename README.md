# vggvox-tf2

A Tensorflow 2.3 Keras reimplementation of the VGGVox network proposed in "VoxCeleb: a large-scale speaker identification dataset"

## Run

Download the dataset, set the correct paths in the notebook and run the notebook

## Results

As many others I did not achieve the claimed *80%* test-accuracy but got to *74%* after a lot of fine-tuning.

## Interesting Findings

1. The validation set should be part of training data to reach better test-accuracy.
2. The dataset is split into two folders, "voxceleb_test" and "voxceleb_train". The data in test must be copied into train before training or else 40 classes are missing.
3. SGD with momentum works better than ADAM by ~10% even though ADAM converges faster during training
4. One can achieve the AvgPooling-Layer with arbitrary input dimensions by using tf.math.reduce_mean on two axis
5. L2 regularization works really well (~6% improvement) while L1+L2 regularization makes test-accuracy worse
6. Learning rate scheduling is important only for SGD (~3% improvement)
7. Loading of .wav files every epoch takes a long time, so parsing them beforehand to numpy arrays actually saves a lot of time during training

