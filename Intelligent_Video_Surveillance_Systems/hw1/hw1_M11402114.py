import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 10)
)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#TensorBoard
writer = SummaryWriter('runs/cifar10_experiment')

epochs = 5
for epoch in range(epochs):
    #Train
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    a=0
    for i, data in enumerate(trainloader, 0):
        a+=1
        inputs, labels = data

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

    #Test
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
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

# 儲存模型
torch.save(model.state_dict(), './linear_net.pth')