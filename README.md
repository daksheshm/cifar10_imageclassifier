
# CIFAR-10 Image Classification using CNNs in PyTorch

This project focuses on building, training, and evaluating Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset using the PyTorch framework. Two different architectures are explored: a custom VGG-style network (TinyVGG) and a ResNet-9 model.

## Project Overview

The primary goal is to accurately classify 32x32 color images into one of 10 distinct classes. The process involves:
1.  **Data Preparation**: Loading the CIFAR-10 dataset and applying a series of augmentations to improve model generalization.
2.  **Model Building**: Implementing two CNN architectures, TinyVGG and ResNet.
3.  **Training**: Training both models on the augmented CIFAR-10 training set, utilizing techniques like learning rate scheduling and gradient clipping.
4.  **Evaluation**: Comparing the performance of the models using loss and accuracy curves and analyzing the final model's predictions with a confusion matrix.

## Dataset

The project uses the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The 10 classes are:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

### Data Augmentation

To make the model more robust and prevent overfitting, the following data augmentations are applied to the training images:
- Random Horizontal Flip
- Random Rotation (20 degrees)
- Color Jitter (brightness, contrast, saturation)
- Random Sharpness Adjustment
- Normalization
- Random Erasing

## Models

Two CNN architectures were implemented and compared:

### 1. TinyVGG
A simple VGG-style network with two convolutional blocks. Each block contains two convolutional layers and a max-pooling layer. The convolutional blocks are followed by a classifier with fully connected layers and dropout for regularization.

### 2. ResNet
A deeper architecture inspired by ResNet, which utilizes residual (or skip) connections to help with the vanishing gradient problem. This allows for deeper networks and often leads to better performance. The implementation consists of several convolutional blocks, some of which are part of residual blocks, followed by a final classifier.

## Training

The models were trained for a total of 60 epochs using the following setup:
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `torch.optim.Adam`
- **Learning Rate Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau` (reduces learning rate when validation loss plateaus)
- **Gradient Clipping**: Used to prevent exploding gradients.
- **Batch Size**: 800

## Results

Both models were trained and their performance was tracked. The loss and accuracy curves demonstrated that the **ResNet-9 model performed significantly better** than the TinyVGG model, achieving higher accuracy and faster convergence.


### Model Evaluation

The final, trained ResNet model was evaluated on the test set. 


The model performs well overall, with an accuracy of approximately 90% on the test set. However, the confusion matrix highlights some class ambiguity, particularly between **cats** and **dogs**, which is a common challenge due to their visual similarities. This could be addressed with a more complex model or more targeted data augmentation.

## How to Run

To run this project, you need a Python environment with the necessary libraries installed.

**Dependencies:**
- PyTorch
- torchvision
- scikit-learn
- pandas
- seaborn
- numpy
- matplotlib
- tqdm
- torchinfo

You can install these using pip:
```bash
pip install torch torchvision scikit-learn pandas seaborn numpy matplotlib tqdm torchinfo
```

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Open and run the notebook:**
    Launch Jupyter Notebook and open `main.ipynb`.
    ```bash
    jupyter notebook main.ipynb
    ```
3.  Execute the cells in order. The notebook will automatically download the CIFAR-10 dataset into a `data/` directory.

**Note:** Training is significantly faster on a machine with a GPU. The notebook is configured to automatically use a CUDA-enabled GPU if one is available.
