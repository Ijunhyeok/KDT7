import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn.functional as F
import os
import cv2 # 이미지 컬러맵 및 오버레이를 위한 OpenCV
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib # 한글 폰트 설정

# --- 설정 및 변수 정의 ---
# TODO: 데이터셋 경로 확인 및 클래스 이름 가져오기 코드를 실제 환경에 맞게 수정하세요.
# 학습 시 사용했던 동일한 이미지 전처리 파이프라인 정의 (모델 입력용)
inference_transform_model_input = transforms.Compose([
    transforms.Resize(256), # 이미지 크기 조절
    transforms.CenterCrop(224), # 중앙 자르기
    transforms.ToTensor(), # 파이토치 텐서로 변환 (0-255 -> 0-1)
    transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.229, 0.224, 0.225]) # 정규화
])

# 클래스 이름 가져오기 (학습 시 사용했던 것과 동일해야 합니다)
# ImageFolder 객체를 임시로 생성하여 클래스 이름을 가져오는 방법 (데이터 폴더가 존재해야 함)
try:
    # ImageFolder는 하위 폴더 이름을 클래스 이름으로 사용합니다. './data/train' 경로를 확인하세요.
    temp_dataset = ImageFolder('./data/train', transform=inference_transform_model_input)
    class_names = temp_dataset.classes # 클래스 이름 (폴더 이름) 리스트
    print(f"데이터셋 폴더에서 클래스 이름 {class_names} 을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"데이터셋 폴더에서 클래스 이름을 불러오는 데 실패했습니다. 오류: {e}")
    print("클래스 이름을 직접 정의하거나, ImageFolder 경로를 수정해주세요.")
    # TODO: 실패 시 실제 학습에 사용된 클래스 이름 리스트로 교체 필요!
    # 예: class_names = ['cheetah_train_resized', 'hyena_train_resized', 'tiger_train_resized']
    class_names = [f'class_{i}' for i in range(3)] # 예시로 3개 클래스 가정 (실제 개수에 맞게 수정)
    print(f"임시 클래스 이름 {class_names} 을(를) 사용합니다.")


# 클래스 이름 매핑 딕셔너리 (모델 출력 이름(폴더 이름)을 원하는 한글 이름으로 변환하기 위해 사용)
# TODO: 실제 class_names 리스트의 항목과 원하는 한글 이름을 여기에 맞게 정의하세요.
class_name_mapping = {
    'cheetah_train_resized': '치타',
    'hyena_train_resized': '하이에나',
    'tiger_train_resized': '호랑이'
    # 예시: '클래스1_폴더이름': '원하는 한글 이름'
}
print(f"클래스 이름 변환 매핑: {class_name_mapping}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # 사용 가능한 경우 GPU, 아니면 CPU 사용
MODEL_PATH = './best_VGG11_model.pt' # 저장된 모델 가중치 파일 경로

# --- 모델 로드 ---
# 모델 구조 정의 (학습 시 사용했던 것과 동일해야 합니다)
# weights=None으로 구조만 불러옵니다. (저장된 가중치를 로드할 것이기 때문)
model = models.vgg11(weights = None)
num_classes = len(class_names) # 데이터셋에서 가져온 클래스 개수
# 최종 분류기 레이어를 학습 시와 동일하게 새로운 선형 레이어로 교체
model.classifier[6] = torch.nn.Linear(4096, num_classes)

# 저장된 모델 가중치 로드
if os.path.exists(MODEL_PATH):
    print(f"모델 가중치 파일 '{MODEL_PATH}'을 로드합니다.")
    # map_location=DEVICE는 저장된 가중치를 지정된 디바이스로 로드합니다.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE) # 모델을 지정된 디바이스로 이동
    model.eval() # 모델을 평가 모드로 설정 (추론 시 필수)
    print("모델 가중치 로드 및 평가 모드 설정 완료.")
else:
    print(f"오류: 모델 가중치 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
    print("모델 파일 경로를 확인하거나, 학습 코드를 다시 실행하여 모델을 저장해주세요.")
    exit() # 파일이 없으면 프로그램 종료

# --- Grad-CAM 함수 정의 ---
def get_grad_cam(model, image_tensor, target_layer, target_category=None):
    """
    주어진 모델, 이미지 텐서, 타겟 레이어에 대한 Grad-CAM 히트맵을 생성합니다.

    Args:
        model (torch.nn.Module): PyTorch 모델.
        image_tensor (torch.Tensor): 모델 입력 이미지 텐서 (배치 차원 포함, ToTensor 및 Normalize 후).
        target_layer (torch.nn.Module): 특징 맵과 그래디언트를 가져올 합성곱 레이어.
        target_category (int, optional): 타겟 클래스 인덱스. None이면 모델의 예측 클래스를 사용합니다.

    Returns:
        numpy.ndarray: [0, 1]로 정규화된 Grad-CAM 히트맵 NumPy 배열 (높이, 너비).
                       훅 실패 또는 그래디언트가 없는 경우 None 반환.
    """
    model.eval() # 모델이 평가 모드인지 다시 확인

    feature_maps = [] # 특징 맵을 저장할 리스트
    gradients = [] # 그래디언트를 저장할 리스트

    # 특징 맵을 저장하기 위한 포워드 훅 함수
    def forward_hook(module, input, output):
        feature_maps.append(output.clone().detach()) # 클론하고 detach하여 그래디언트 계산에서 제외
        # output은 [Batch, Channels, H, W] 형태

    # 그래디언트를 저장하기 위한 백워드 훅 함수
    def backward_hook(module, grad_input, grad_output):
        # grad_output[0]는 해당 모듈 출력에 대한 손실의 그래디언트
        gradients.append(grad_output[0].clone().detach()) # 클론하고 detach
        # grad_output[0]은 [Batch, Channels, H, W] 형태

    # 타겟 레이어에 훅 등록
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # 포워드 패스 실행
    # image_tensor는 이미 to(DEVICE) 되어 있어야 함
    output = model(image_tensor) # output은 로짓 [Batch, num_classes] 형태

    # 타겟 클래스 결정 (지정되지 않았다면 예측 클래스 사용)
    if target_category is None:
        target_category = torch.argmax(output, dim=1).item()

    # 모델의 모든 그래디언트 초기화
    model.zero_grad()

    # 타겟 클래스 로짓에 대한 그래디언트 계산
    # 타겟 클래스 인덱스에만 1이고 나머지는 0인 텐서 생성
    one_hot = torch.zeros_like(output).to(image_tensor.device)
    one_hot[0][target_category] = 1 # 배치 크기가 1이라고 가정

    # 백워드 패스 실행 (타겟 로짓에 대한 그래디언트 계산 시작)
    # retain_graph=True는 이후에도 그래디언트가 필요할 경우 사용하지만, 여기서는 필요 없을 수 있습니다.
    output.backward(gradient=one_hot, retain_graph=True)

    # 특징 맵과 그래디언트가 제대로 캡처되었는지 확인
    if not feature_maps or not gradients:
        print("오류: 특징 맵 또는 그래디언트 캡처에 실패했습니다. 타겟 레이어를 확인하세요.")
        forward_handle.remove() # 훅 해제
        backward_handle.remove() # 훅 해제
        return None

    # 캡처된 데이터 가져오기 (훅당 한 번의 캡처를 가정)
    # features_map[0] : [1, Channels, H_feature, W_feature]
    # gradients[0] : [1, Channels, H_feature, W_feature]
    feature_map = feature_maps[0]
    gradient = gradients[0]

    # 사용 후 훅 해제 (메모리 누수 방지)
    forward_handle.remove()
    backward_handle.remove()

    # 가중치 계산 (각 채널의 공간적 평균 그래디언트)
    # weights shape: [1, Channels, 1, 1]
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)

    # Grad-CAM 계산: 특징 맵과 가중치를 곱하고 채널 차원으로 합산
    # cam shape: [1, H_feature, W_feature]
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True) # 보간을 위해 keepdim=True 유지

    # ReLU 적용 (양수 기여만 고려)
    cam = F.relu(cam) # Shape: [1, 1, H_feature, W_feature]

    # CAM 크기를 모델 입력 이미지의 공간적 크기 (예: 224x224)로 보간
    # input_h, input_w는 Resize/CenterCrop 후의 높이/너비입니다.
    input_h, input_w = image_tensor.shape[2], image_tensor.shape[3]
    cam = F.interpolate(cam, size=(input_h, input_w), mode='bilinear', align_corners=False) # Shape: [1, 1, input_h, input_w]

    # CAM을 [0, 1] 범위로 정규화
    cam = cam.squeeze(0).squeeze(0) # 배치 및 채널 차원 제거 -> Shape: [input_h, input_w]
    # CAM 값이 모두 같거나 0인 경우 (예: 학습이 전혀 안 된 경우) 예외 처리
    if cam.max() == cam.min():
         print("경고: Grad-CAM 히트맵 값이 모두 동일합니다 (모델이 특징을 구분하지 못할 수 있습니다).")
         return np.zeros((input_h, input_w), dtype=np.float32) # 검은색 히트맵 반환

    cam = (cam - cam.min()) / (cam.max() - cam.min()) # [0, 1]로 정규화

    return cam.cpu().numpy() # NumPy 배열로 변환하여 반환

# --- 히트맵 오버레이 헬퍼 함수 ---
def overlay_heatmap(original_img_pil, heatmap_np):
    """
    원본 PIL 이미지 위에 히트맵을 겹쳐서 시각화합니다.

    Args:
        original_img_pil (PIL.Image.Image): 원본 이미지.
        heatmap_np (numpy.ndarray): [0, 1]로 정규화된 히트맵 NumPy 배열 [높이, 너비].
                                   모델 입력 이미지의 공간 크기와 일치해야 합니다.

    Returns:
        PIL.Image.Image: 히트맵이 오버레이된 이미지.
    """
    # 원본 PIL 이미지를 NumPy 배열 (높이, 너비, 채널)로 변환 - RGB 형태
    original_img_np = np.array(original_img_pil)

    # 히트맵 크기를 원본 PIL 이미지 크기로 조정
    original_h, original_w = original_img_np.shape[0], original_img_np.shape[1]
    # cv2.resize는 너비, 높이 순서 (np.shape는 높이, 너비 순서)
    heatmap_resized = cv2.resize(heatmap_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    # 히트맵에 컬러맵 적용 (예: cv2.COLORMAP_JET 또는 cv2.COLORMAP_HOT)
    # 컬러맵 적용 전 히트맵 값을 0-255 범위의 uint8로 변환
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # 원본 이미지 NumPy 배열을 RGB에서 BGR로 변환 (OpenCV는 기본적으로 BGR 사용)
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    # addWeighted를 사용하여 히트맵 오버레이 (혼합)
    # 결과 = 원본_이미지 * alpha + 컬러맵_히트맵 * beta + gamma
    # alpha: 원본 이미지 가중치, beta: 히트맵 가중치
    overlayed_img_bgr = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored, 0.4, 0) # alpha, beta 값 조정 가능

    # Matplotlib 또는 PIL 표시를 위해 다시 BGR에서 RGB로 변환
    overlayed_img_rgb = cv2.cvtColor(overlayed_img_bgr, cv2.COLOR_BGR2RGB)

    # PIL Image 객체로 변환하여 반환
    return Image.fromarray(overlayed_img_rgb)

# --- 메인 실행 부분 ---

# TODO: 분류할 새로운 이미지 파일 경로를 여기에 입력하세요.
new_image_path = './test/cheetah.jpg' # 실제 이미지 파일 경로로 변경해야 합니다!

# 이미지 파일이 존재하는지 확인
if not os.path.exists(new_image_path):
    print(f"오류: 이미지 파일 '{new_image_path}'을 찾을 수 없습니다.")
else:
    try:
        print(f"'{new_image_path}' 이미지 분류 및 Grad-CAM 생성 시작...")

        # 1. 원본 이미지 로드
        original_image_pil = Image.open(new_image_path).convert('RGB')
        print("원본 이미지 로드 완료.")

        # 2. 모델 입력용 이미지 전처리 및 디바이스 이동
        image_tensor = inference_transform_model_input(original_image_pil).unsqueeze(0).to(DEVICE)
        print("이미지 전처리 및 디바이스 이동 완료.")

        # 3. 이미지 분류 수행
        with torch.no_grad():
            output = model(image_tensor) # 로짓 출력

        probs = F.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probs).item()
        original_predicted_class_name = class_names[predicted_index]
        predicted_korean_name = class_name_mapping.get(original_predicted_class_name, original_predicted_class_name)

        print("\n--- 분류 결과 ---")
        print(f"예측 클래스: {predicted_korean_name}")
        print("각 클래스 확률:")
        for i, prob in enumerate(probs.cpu().numpy()):
            original_class_name = class_names[i]
            korean_class_name = class_name_mapping.get(original_class_name, original_class_name)
            print(f"  {korean_class_name}: {prob:.4f}")
        print("----------------\n")

        # 4. Grad-CAM 생성
        # Grad-CAM을 적용할 타겟 레이어 지정
        # VGG11의 features 시퀀셜 모듈에서 원하는 Conv2d 레이어를 선택합니다.
        # model.features[인덱스] 형태로 접근합니다.
        # print(model)을 실행하여 정확한 인덱스를 확인하는 것이 좋습니다.
        #
        # 예시:
        # 타겟 레이어를 마지막 Conv2d 레이어 (인덱스 18)로 설정
        # target_layer = model.features[18]

        # 타겟 레이어를 두 번째 Conv2d 레이어 (인덱스 3)로 설정
        # target_layer = model.features[3]
        
        """
        (0): Conv2d (...)
        (1): ReLU (...)
        (2): MaxPool2d (...)
        (3): Conv2d (...)
        (4): ReLU (...)
        (5): MaxPool2d (...)
        (6): Conv2d (...)
        (7): ReLU (...)
        (8): Conv2d (...)
        (9): ReLU (...)
        (10): MaxPool2d (...)
        (11): Conv2d (...)
        (12): ReLU (...)
        (13): Conv2d (...)
        (14): ReLU (...)
        (15): MaxPool2d (...)
        (16): Conv2d (...)
        (17): ReLU (...)
        (18): Conv2d (...)  <-- 일반적으로 마지막 Conv2d 레이어 (인덱스 18)
        (19): ReLU (...)
        (20): MaxPool2d (...)
        """

        # 타겟 레이어를 중간의 Conv2d 레이어 (인덱스 13)로 설정
        target_layer_index = 3 # 여기에 원하는 Conv2d 레이어의 인덱스를 입력하세요.
        target_layer = model.features[target_layer_index]


        print(f"Grad-CAM 생성을 위한 타겟 레이어: {target_layer.__class__.__name__} (features[{target_layer_index}])")

        # 예측된 클래스(predicted_index)에 대한 Grad-CAM 히트맵 생성
        heatmap = get_grad_cam(model, image_tensor, target_layer, target_category=predicted_index)
        print("Grad-CAM 히트맵 생성 완료.")

        # 5. 히트맵 시각화
        if heatmap is not None:
            # 원본 이미지 위에 생성된 히트맵 오버레이
            overlayed_image = overlay_heatmap(original_image_pil, heatmap)
            print("히트맵 오버레이 이미지 생성 완료.")

            # 결과 이미지 표시
            print("결과 이미지를 표시합니다.")
            plt.figure(figsize=(8, 8))
            plt.imshow(overlayed_image)
            plt.title(f'Grad-CAM for Predicted Class: {predicted_korean_name}\nTarget Layer: features[{target_layer_index}]') # 제목에 레이어 인덱스 추가
            plt.axis('off')
            plt.show()
        else:
            print("Grad-CAM 히트맵 생성 과정에 문제가 발생했습니다.")

    except Exception as e:
        print(f"코드 실행 중 예상치 못한 오류 발생: {e}")