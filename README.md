# pytorch_tutorial🔖

<p align='center'>
    <a href="https://pytorch.org/"> 
        <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg"> 
    </a>
</p>

## [图像分类](./image_classification)

学习图像分类前的准备：

* [pytorch基础](./image_classification/pytorch_basics.ipynb)
* [线性回归](./image_classification/linear_regression.ipynb)

### MNIST

如果有人数据集能够无障碍的下载MNIST数据集，那么：

```python
# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='data/MNIST', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='data/MNIST', train=False, transform=torchvision.transfroms.ToTensor(), download=True)
```

如果不能，先从MNIST数据集官网上把数据集下载下来, 存在'data/MNIST/'中（data与项目文件同级），再使用这个数据集加载类进行操作。

```python
import numpy as np
import gzip
import os
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        # The file name prefix is obtained according to whether it is a training set or not.
        self.file_pre = 'train' if train == True else 't10k'
        self.transform = transform

        # Generate the image and label file path of the corresponding dataset.
        self.label_path = os.path.join(root, '%s-labels-idx1-ubyte.gz' % self.file_pre)
        self.image_path = os.path.join(root, '%s-images-idx3-ubyte.gz' % self.file_pre)

        # Read file data and return pictures and labels.
        self.images, self.labels = self.__read_data__(self.image_path, self.label_path)

    def __read_data__(self, image_path, label_path):
        # Data set reading.
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)
        return images, labels

    def __getitem__(self, index):
        image, label = self.images[index], int(self.labels[index])

        # If you need to convert to tensor, use tansform.
        if self.transform is not None:
            image = self.transform(np.array(image))  # Avoid bug: use np.array
        return image, label

    def __len__(self):
        return len(self.labels)
```

加载方式与第一种情况相同，但是少了download参数。

#### [逻辑回归模型](./image_classification/logistic_regression.ipynb)

目前准确率：92.17%

#### [前向传播神经网络模型](./image_classification/feedforward_neural_network.ipynb)

目前准确率：97.16%

#### [简单卷积神经网络](./image_classification/convolutional_neural_network.ipynb)

目前准确率：98.8%

#### [LeNet-5](./image_classification/lenet-5.ipynb)

目前准确率：99.04%

#### [RNN](./image_classification/recurrent_neural_network.ipynb)

目前准确率：97.01%

### [CIFAR10](./image_classification/comparison.ipynb)

#### [AlexNet](./image_classification/alexnet.ipynb)

目前准确率：86.1%

#### [VGGNet](./image_classification/vggnet.ipynb)

VGGNet模型总的来说，分为VGG16和VGG19两类，区别在于模型的层数不同，以下'M'参数代表池化层，数据代表各层滤波器的数量。

```python
# Define VGG-16 and VGG-19.
cfg = {
    'VGG-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
    'VGG-19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M']
}
```

目前准确率：92.23%

`VGG-19`

目前准确率：91.99%

#### [GoogLeNet](./image_classification/googlenet.ipynb)

```python
GoogLeNet(num_classes, aux_logits, init_weights)
```

如果开启辅助分类器，那么`aux_logits=True`，目前准确率：86.99%；如果不开启，那么`aux_logits=False`，目前准确率：85.88%。

#### [ResNet](./image_classification/resnet.ipynb)

目前准确率：89.89%

## [目标检测](./object_detection)

### 一阶段算法

#### [YOLOv5s](./object_detection/video_detection.ipynb)

### 二阶段算法

## 参考资料

* 神经网络与深度学习：邱锡鹏著
* [Jack-Cherish/Deep-Learning](https://github.com/Jack-Cherish/Deep-Learning)

## LICENSE
[MIT LICENSE](./LICENSE)