import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time  # 시간 측정을 위한 모듈 추가
import datetime  # 시간 형식화를 위한 모듈 추가

# 데이터 준비
IMG_ROOT = './새 폴더/upimage/'  # 경로 수정 필요

preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터셋 생성
try:
    imgDS = ImageFolder(root=IMG_ROOT, transform=preprocessing)
    # ImageFolder 작동 시
    
    # 데이터 분리
    indices = list(range(len(imgDS)))
    labels = [imgDS[i][1] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    train_dataset = Subset(imgDS, train_indices)
    test_dataset = Subset(imgDS, test_indices)
    
    num_classes = len(imgDS.classes)
    
except:
    # CSV 파일을 사용하는 방식으로 변경
    import pandas as pd
    from PIL import Image
    import os
    from torch.utils.data import Dataset
    
    class BirdDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            self.data_info = pd.read_csv(csv_file, header=None, names=['img_path', 'upscale_img_path', 'label'])
            self.img_dir = img_dir
            self.transform = transform
            
            self.label_set = sorted(list(set(self.data_info['label'])))
            self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
            self.data_info['label_idx'] = self.data_info['label'].apply(lambda x: self.label_to_idx[x])
            
            self.classes = self.label_set
        
        def __len__(self):
            return len(self.data_info)
        
        def __getitem__(self, idx):
            img_name = os.path.join(self.img_dir, os.path.basename(self.data_info.iloc[idx, 1]))  # upscale_img_path 사용
            image = Image.open(img_name).convert('RGB')
            label = self.data_info.iloc[idx, 3]  # label_idx 사용
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    # CSV 파일 및 이미지 경로 설정
    csv_file = './새 폴더/filtered_dataset.csv'  # CSV 파일 경로 수정 필요
    
    # 데이터셋 생성
    dataset = BirdDataset(csv_file=csv_file, img_dir=IMG_ROOT, transform=preprocessing)
    
    # 데이터 분리
    indices = list(range(len(dataset)))
    labels = [dataset.data_info.iloc[i]['label_idx'] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    num_classes = len(dataset.classes)

# CNN 모델 정의
class BirdCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 특징 추출 부분 - Sequential 사용
        self.features = nn.Sequential(
            # 첫 번째 합성곱 블록
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 두 번째 합성곱 블록
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 평탄화
            nn.Flatten()
        )
        
        # 크기 자동 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_output = self.features(dummy_input)
            flattened_size = dummy_output.shape[1]
            print(f"자동 계산된 특징 크기: {flattened_size}")
        
        # 분류 부분 - Sequential 사용
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 특징 추출
        features = self.features(x)
        
        # 분류
        out = self.classifier(features)
        
        return out


# 학습 파라미터 설정
num_epochs = 25
batch_size = 32
learning_rate = 0.0005
PAT_CNT = 5		# 모델 성능 미개선 횟수 체크
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
SCHEDULER   = ReduceLROnPlateau(optimizer, mode='min', patience=PAT_CNT)

# 모델 파라미터 수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 학습 파라미터 출력
print("\n" + "="*50)
print("학습 파라미터 정보:")
print("="*50)
print(f"에포크 수: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"학습률: {learning_rate}")
print(f"옵티마이저: Adam")
print(f"손실 함수: CrossEntropyLoss")
print(f"훈련 데이터 크기: {len(train_dataset)}")
print(f"테스트 데이터 크기: {len(test_dataset)}")
print(f"모델 파라미터 수: {total_params:,}")
print(f"사용 장치: {device}")
print("="*50 + "\n")

# 학습 함수
def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch_num):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        batch_start_time = time.time()
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        ''' 너무 많은 프린트 싫으면 이 부분 주석 처리'''
        # # 진행 상황 출력 (10배치마다)
        # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
        #     print(f'에포크 {epoch_num+1}/{num_epochs} | 배치 {batch_idx+1}/{len(data_loader)} | '
        #           f'손실: {loss.item():.4f} | 배치 처리 시간: {batch_time:.2f}초')
    
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    
    return running_loss / len(data_loader), 100. * correct / total, epoch_time

# 평가 함수
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    
    return running_loss / len(data_loader), 100. * correct / total, eval_time

# 시간 형식화 함수
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# 학습 실행
best_acc = 0.0
best_model_path = 'best_bird_model.pth'

train_losses, train_accs = [], []
test_losses, test_accs = [], []

total_start_time = time.time()

# 조기종료 위한 기준값 저장 변수
EARLY_STOP = 3

print("학습 시작...")
for epoch in range(num_epochs):
    epoch_start = time.time()
    train_loss, train_acc, train_time = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
    test_loss, test_acc, test_time = evaluate(model, test_loader, loss_fn, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print("\n" + "-"*70)
    print(f'에포크 {epoch+1}/{num_epochs} 결과:')
    print(f'훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.2f}%')
    print(f'테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_acc:.2f}%')
    print(f'훈련 시간: {format_time(train_time)} ({train_time:.2f}초)')
    print(f'평가 시간: {format_time(test_time)} ({test_time:.2f}초)')
    
    # 경과 시간 계산 및 출력
    elapsed_time = time.time() - total_start_time
    estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
    remaining_time = estimated_total - elapsed_time
    
    print(f'총 경과 시간: {format_time(elapsed_time)} ({elapsed_time:.2f}초)')
    print(f'예상 남은 시간: {format_time(remaining_time)} ({remaining_time:.2f}초)')
    
    # 현재 모델이 이전에 저장된 최고 모델보다 성능이 좋으면 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'새로운 최고 성능 모델이 저장되었습니다! 정확도: {best_acc:.2f}%')
    print("-"*70 + "\n")

    # # 최저 손실값을 찾기 위한 learning rate를 알아서 조정한다.
    # SCHEDULER.step(test_loss)

        # 조기종료 체크
    if SCHEDULER.num_bad_epochs >= SCHEDULER.patience:
        EARLY_STOP -= 1
    
    if not EARLY_STOP:
        print(f'{epoch}-EPOCHS : 성능 개선이 없어서 조기 종료합니다.')

total_end_time = time.time()
total_training_time = total_end_time - total_start_time

print(f'학습 완료!')
print(f'총 학습 시간: {format_time(total_training_time)} ({total_training_time:.3f}초)')
print(f'최종 최고 성능 모델 정확도: {best_acc:.3f}%')

# 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('training_results.png')  # 결과 저장
plt.show()

# 훈련 결과 요약 출력
print("\n" + "="*50)
print("훈련 결과 요약:")
print("="*50)
print(f"에포크 수: {num_epochs}")
print(f"배치 크기: {batch_size}")
print(f"학습률: {learning_rate}")
print(f"최종 훈련 손실: {train_losses[-1]:.4f}")
print(f"최종 훈련 정확도: {train_accs[-1]:.2f}%")
print(f"최종 테스트 손실: {test_losses[-1]:.4f}")
print(f"최종 테스트 정확도: {test_accs[-1]:.2f}%")
print(f"최고 테스트 정확도: {best_acc:.2f}%")
print(f"총 훈련 시간: {format_time(total_training_time)}")
print(f"평균 에폭크 시간: {format_time(total_training_time/num_epochs)}")
print("="*50)

# print(model)



# # itertools 모듈에서 product 임포트
# from itertools import product  # 하이퍼파라미터 조합을 생성하기 위해 필요

# # 하이퍼파라미터 탐색 로직
# param_grid = {
#     'learning_rate': [0.00005, 0.0001, 0.0005
#                       ],
#     'batch_size': [8, 16, 32],
#     'dropout_rate': [0.3, 0.4, 0.5],
#     'num_epochs': [20, 25, 30]  # 에포크 탐색 범위 추가
# }

# best_params = None
# best_accuracy = 0.0

# for lr, batch_size, dropout_rate, num_epochs in product(
#     param_grid['learning_rate'],
#     param_grid['batch_size'],
#     param_grid['dropout_rate'],
#     param_grid['num_epochs']
# ):
#     print(f"\nTesting: LR={lr}, Batch={batch_size}, Dropout={dropout_rate}, Epochs={num_epochs}")
    
#     # 데이터 로더 정의
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
#     # BirdCNN 초기화 전에 flattened_size 계산
#     with torch.no_grad():
#         dummy_input = torch.zeros(1, 3, 224, 224)  # 입력 샘플
#         features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten()
#         )
#         dummy_output = features(dummy_input)
#         flattened_size = dummy_output.shape[1]  # 동적으로 계산
    
#     # BirdCNN 모델 초기화
#     class BirdCNN(nn.Module):
#         def __init__(self, dropout_rate=0.3):
#             super().__init__()
#             self.features = features
#             self.classifier = nn.Sequential(
#                 nn.Linear(flattened_size, 128),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_rate),
#                 nn.Linear(128, num_classes)
#             )
        
#         def forward(self, x):
#             features = self.features(x)
#             return self.classifier(features)

#     model = BirdCNN(dropout_rate).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     # 여러 에폭 동안 학습 및 평가
#     avg_test_acc = 0.0  # 테스트 정확도의 평균을 저장
#     for epoch in range(num_epochs):
#         train_loss, train_acc, _ = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch_num=epoch)
#         test_loss, test_acc, _ = evaluate(model, test_loader, loss_fn, device)
#         avg_test_acc += test_acc  # 테스트 정확도를 누적
        
#         # 진행 상태 출력
#         print(f"Epoch {epoch+1}/{num_epochs}: Train Accuracy={train_acc:.2f}%, Test Accuracy={test_acc:.2f}%")
    
#     avg_test_acc /= num_epochs  # 에폭 수로 나눠 평균 계산
#     print(f"Average Test Accuracy for LR={lr}, Batch={batch_size}, Dropout={dropout_rate}, Epochs={num_epochs}: {avg_test_acc:.2f}%")
    
#     # 성능 비교 후 최적의 파라미터 저장
#     if avg_test_acc > best_accuracy:
#         best_accuracy = avg_test_acc
#         best_params = {'learning_rate': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate, 'num_epochs': num_epochs}
#         print(f"New best params: {best_params} with average accuracy: {best_accuracy:.2f}%")

# # 최적의 파라미터 출력
# print("\n하이퍼파라미터 탐색 완료!")
# print(f"최적의 학습률: {best_params['learning_rate']}")
# print(f"최적의 배치 크기: {best_params['batch_size']}")
# print(f"최적의 드롭아웃 비율: {best_params['dropout_rate']}")
# print(f"최적의 에포크 수: {best_params['num_epochs']}")
# print(f"최고 평균 테스트 정확도: {best_accuracy:.2f}%")