import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 匯入進度條套件

# ==========================================
# 0. 環境與全域設定
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 30
SAVE_DIR = "./model"
LOG_DIR = "./runs"

# 定義自定義 Dataset 類別
class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, hf_subset, transform=None):
        self.subset = hf_subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 1. 訓練與評估函數（已加入 tqdm 進度條）
# ==========================================
def train_and_evaluate(model_name, model_tag, train_loader, val_loader):
    print(f"\n==================== 開始訓練: {model_name} ====================")
    
    # 載入預訓練模型並重置分類頭
    model = timm.create_model(model_tag, pretrained=True, num_classes=NUM_CLASSES)
    
    # 凍結 backbone 參數
    for param in model.parameters():
        param.requires_grad = False
        
    model.reset_classifier(num_classes=NUM_CLASSES)
    
    # 僅解凍最後分類層
    if hasattr(model, 'fc') and model.fc is not None:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier') and model.classifier is not None:
        for param in model.classifier.parameters():
            param.requires_grad = True

    model = model.to(DEVICE)
    
    # 保持 BatchNorm 評估模式
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable_params = count_trainable_parameters(model)
    print(f"可訓練參數數量 (Trainable Parameters): {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, model_name))

    best_val_acc = 0.0
    best_val_loss = float('inf')
    final_train_acc = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # --- 訓練階段 ---
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 使用 tqdm 包裝 train_loader，打造動態進度條
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train")
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            # 即時在進度條後方更新目前的 Batch Loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- 驗證階段 ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # 使用 tqdm 包裝 val_loader
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val  ")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # 儲存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}_best.pth"))

        final_train_acc = epoch_train_acc

        # 寫入 TensorBoard
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", epoch_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch + 1)
        writer.add_scalar("Accuracy/Validation", epoch_val_acc, epoch + 1)

        # 每個 Epoch 結束後列印總結
        print(f"👉 [ Epoch {epoch+1} 總結 ] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%\n")

    training_time = time.time() - start_time
    writer.close()

    return {
        "model_name": model_name,
        "final_train_acc": final_train_acc,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "training_time": training_time,
        "trainable_params": trainable_params
    }


# ==========================================
# 2. 主程式入口（Windows 多行程保護）
# ==========================================

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"使用訓練裝置: {DEVICE}")
print("正在載入資料集...")
#載入資料集
dataset = load_dataset("Hemg/Brain-Tumor-MRI-Dataset")
train_data_full = dataset["train"]
# 切分資料集(分成train跟valid)
train_val_test = train_data_full.train_test_split(test_size=0.2, seed=42)
train_val_subset = train_val_test["train"]
test_subset = train_val_test["test"]
split_dataset = train_val_subset.train_test_split(test_size=0.2, seed=42)
train_subset = split_dataset["train"]
val_subset = split_dataset["test"]
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = BrainTumorDataset(train_subset, transform=transform)
val_dataset = BrainTumorDataset(val_subset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 執行模型訓練
results = []
results.append(train_and_evaluate("ResNet50", "resnet50.a1_in1k", train_loader, val_loader))
results.append(train_and_evaluate("MobileNetV3", "mobilenetv3_large_100.ra_in1k", train_loader, val_loader))

# 輸出最終比較報告
print("\n" + "="*50)
print("Part 5: 效能比較總結報告 (Performance Comparison)")
print("="*50)

for r in results:
    print(f"模型名稱 (Model): {r['model_name']}")
    print(f" - 最終訓練準確率: {r['final_train_acc']*100:.2f}%")
    print(f" - 最佳驗證準確率: {r['best_val_acc']*100:.2f}%")
    print(f" - 對應驗證損失: {r['best_val_loss']:.4f}")
    print(f" - 訓練總耗時: {r['training_time']:.2f} 秒")
    print(f" - 可訓練參數數量: {r['trainable_params']:,}")
    print("-" * 50)