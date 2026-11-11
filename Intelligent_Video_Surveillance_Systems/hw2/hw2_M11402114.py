import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

#用gpu進行訓練
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用中設備: {device}")
# ==================== 1. 影像增強與前處理 (Requirement 1 & 2) ====================
# 依照作業要求加入 Image Augmentation (例如 RandomCrop, RandomHorizontalFlip) 
# 與 Preprocessing (計算並標準化/減去 mean)
transform_train = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomCrop(32, padding=4),#隨機裁切
    v2.RandomHorizontalFlip(),#隨機水平翻轉
    # CIFAR-10 的約略 Mean 與 Std 進行正規化 (減去 mean 的概念)
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==================== 2. 模型結構調整 (Requirement 3 & 4) ====================
# 加入 ReLU 激活函數與 Batch Normalization 層
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 512),
    nn.BatchNorm1d(512),       # Batch Normalization (Requirement 4)
    nn.ReLU(),                 # ReLU 激活函數 (Requirement 3)
    nn.Linear(512, 10)
).to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

#依序使用 Adam、SGD、AdamW 優化器進行訓練並產生三個不同的 TensorBoard 實驗結果
opt = (optim.Adam(model.parameters(), lr=0.001),
optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01))

wri = (SummaryWriter('runs/cifar10_adam_experiment_withAugmentation'),
SummaryWriter('runs/cifar10_SGD_experiment_withAugmentation'),
SummaryWriter('runs/cifar10_adamw_experiment_withAugmentation'))

for optimizer,writer in zip(opt,wri):
    epochs = 30
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 計算訓練集正確數
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train

        # Test
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / len(testloader)
        epoch_test_acc = 100 * correct_test / total_test

        # 將四個指標寫入 TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/test', epoch_test_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/test', epoch_test_acc, epoch)

        print(f'Epoch [{epoch+1}/{epochs}] | '
              f'Train Loss: {epoch_train_loss:.3f} | Train Acc: {epoch_train_acc:.2f}% | '
              f'Test Loss: {epoch_test_loss:.3f} | Test Acc: {epoch_test_acc:.2f}%')

    writer.close()
    print('Finished Training & Evaluation')