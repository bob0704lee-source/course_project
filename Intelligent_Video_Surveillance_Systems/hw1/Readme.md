Homework 1: PyTorch Image Classification with CIFAR-10
 

Deadline 10/02 Friday 17:00

Objective
Learn the basic PyTorch workflow:

Dataset → DataLoader → Model → Loss → Optimizer → Training → Testing → TensorBoard

1. Dataset
Use the CIFAR-10 dataset provided by torchvision.

CIFAR-10 contains 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Each image has a size of:

3 × 32 × 32

Use:

torchvision.datasets.CIFAR10

Create:

Training dataset

Test dataset

Training DataLoader

Test DataLoader

Recommended:

batch_size = 64

2. Model
Do not use a CNN.

Use only a simple linear classifier:

model = nn.Sequential(

    nn.Flatten(),

    nn.Linear(3 * 32 * 32, 10)

)

The model structure is:

CIFAR-10 Image

      ↓

3 × 32 × 32

      ↓

Flatten

      ↓

3072 Values

      ↓

Linear(3072, 10)

      ↓

10 Class Scores

3. Loss Function
Use Cross-Entropy Loss:

criterion = nn.CrossEntropyLoss()

4. Optimizer
Use SGD or Adam.

For example:

optimizer = torch.optim.SGD(

    model.parameters(),

    lr=0.01

)

5. Training
Train the linear classifier using the CIFAR-10 training dataset.

6. Testing
Evaluate the trained model using the CIFAR-10 test dataset.

7. TensorBoard
Use TensorBoard to display the following four curves:

Loss/train

Loss/test

Accuracy/train

Accuracy/test

Tensorboard tutorial 

Important
Do not use CNN, ResNet, pretrained models, or other advanced neural networks in this homework.

The purpose of this homework is to understand the basic PyTorch training and testing procedure.

The official PyTorch CIFAR-10 tutorial shows that CIFAR-10 images are 3-channel, 32 × 32 images and demonstrates the standard PyTorch workflow for dataset loading, training, and testing.