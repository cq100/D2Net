# D2Net

Code for this paper [An LED Detection and Recognition Method Based on Deep Learning in Vehicle Optical Camera Communication]

XU SUN,  QING CHENG


## Training

#### Command

```python train.py```

This configuration file is core/config.py



## Testing

#### Command

To test on a single image,

```python detect.py --image ./data/1.jpg```


## Pre-trained model

The pre-trained model is "D2Net.h5". It can be downloaded via the links below:
- [D2Net.h5](https://drive.google.com/file/d/11rkp2p7WjG1JD2wXzWsXAl6nOKJ8Oiap/view?usp=sharing)

## Remarks

1. When you only have one GPU, please comment out the Gradnorm code block
2. The pre-trained model is trained on a private data set. When you try to do other tasks, please retrain the model on a new data set
3. Please set the learning rate of the discriminator reasonably according to your task
4. In order to facilitate understanding, we simplified this code. If you have questions, please contact us in time
5. This code will continue to be updated


```

