import timm
import torch
import os
# 設定類別數量（對應您的 4 個腦瘤分類）
NUM_CLASSES = 4
SAVE_DIR = "./model"
# 確保本地端存在該資料夾
os.makedirs(SAVE_DIR, exist_ok=True)
# 1. 載入 Model A: ResNet-50 (timm/resnet50.a1_in1k)
# pretrained=True 會自動從 Hugging Face / timm 權重庫下載預訓練參數
model_a = timm.create_model(
    'resnet50.a1_in1k', 
    pretrained=True, 
    num_classes=NUM_CLASSES
)

# 2. 載入 Model B: MobileNetV3-Large (timm/mobilenetv3_large_100.ra_in1k)
model_b = timm.create_model(
    'mobilenetv3_large_100.ra_in1k', 
    pretrained=True, 
    num_classes=NUM_CLASSES
)

# 3. 將模型權重儲存到本地端的 ./model 資料夾中
# 實務上通常儲存 state_dict（權重字典）即可，檔案較小且安全
resnet_path = os.path.join(SAVE_DIR, "resnet50_brain_tumor.pth")
mobilenet_path = os.path.join(SAVE_DIR, "mobilenetv3_brain_tumor.pth")

torch.save(model_a.state_dict(), resnet_path)
torch.save(model_b.state_dict(), mobilenet_path)

print(f"模型已成功儲存至：")
print(f" - {resnet_path}")
print(f" - {mobilenet_path}")