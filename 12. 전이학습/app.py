# app.py (한국어 번역)

# 필요한 라이브러리 임포트
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import cv2 # 이미지 컬러맵 및 오버레이를 위한 OpenCV
import numpy as np
import torch.nn as nn

# Flask 웹 프레임워크 임포트
from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO # 이미지 데이터를 메모리에서 처리하기 위한 BytesIO
import base64 # 이미지를 base64 문자열로 인코딩하기 위한 라이브러리

# Flask 애플리케이션 인스턴스 생성
app = Flask(__name__)

# --- 모델 로드 (애플리케이션 시작 시 한 번만 로드) ---
# ... (모델 로드 부분은 이전 코드와 동일) ...
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './best_VGG11_model.pt' # 실제 경로로 수정 필요

model = models.vgg11(weights = None)

num_classes = 3 # 예시: 클래스 개수
class_names = ['cheetah_train_resized', 'hyena_train_resized', 'tiger_train_resized'] # 예시

class_name_mapping = {
    'cheetah_train_resized': '치타',
    'hyena_train_resized': '하이에나',
    'tiger_train_resized': '호랑이'
}

try:
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
except Exception as e:
    print(f"모델 분류기 레이어 수정 중 오류 발생: {e}. 모델 구조를 확인하세요.")
    model = None

if model is not None and os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("모델 가중치 로드 및 평가 모드 설정 완료.")
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}. 모델 파일이 올바른지 확인하세요.")
        model = None
else:
    if model is not None:
         print(f"오류: 모델 가중치 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
    if model is not None:
        print("모델 로드 상태 확인 필요.")
    model = None


# --- 이미지 전처리 함수 정의 (모델 입력용) ---
inference_transform_model_input = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], std=[0.229, 0.224, 0.225])
])

# --- Grad-CAM 및 오버레이 함수 정의 ---
# (get_grad_cam 함수는 이전 코드의 XGrad-CAM 변형 또는 표준 Grad-CAM 함수 사용)
# (overlay_heatmap 함수는 이전 코드와 동일)
def get_grad_cam(model, image_tensor, target_layer, target_category=None):
    # ... (get_grad_cam 함수 내용 복사) ...
    model.eval() # 모델이 평가 모드인지 다시 확인

    feature_maps = [] # 특징 맵을 저장할 리스트
    gradients = [] # 그래디언트를 저장할 리스트

    def forward_hook(module, input, output):
        feature_maps.append(output.clone().detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].clone().detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)

    if target_category is None:
        target_category = torch.argmax(output, dim=1).item()

    model.zero_grad()

    one_hot = torch.zeros_like(output).to(image_tensor.device)
    one_hot[0][target_category] = 1

    output.backward(gradient=one_hot, retain_graph=True)

    if not feature_maps or not gradients:
        print("오류: 특징 맵 또는 그래디언트 캡처에 실패했습니다. 타겟 레이어를 확인하세요.")
        forward_handle.remove()
        backward_handle.remove()
        return None

    feature_map = feature_maps[0]
    gradient = gradients[0]

    forward_handle.remove()
    backward_handle.remove()

    # 가중치 계산 (XGrad-CAM 변형 방식 사용)
    weights = torch.mean(gradient * feature_map, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = F.relu(cam)

    input_h, input_w = image_tensor.shape[2], image_tensor.shape[3]
    cam = F.interpolate(cam, size=(input_h, input_w), mode='bilinear', align_corners=False)

    cam = cam.squeeze(0).squeeze(0)
    if cam.max() == cam.min():
         print("경고: Grad-CAM 히트맵 값이 모두 동일합니다 (모델이 특징을 구분하지 못할 수 있습니다).")
         return np.zeros((input_h, input_w), dtype=np.float32)

    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam.cpu().numpy()

def overlay_heatmap(original_img_pil, heatmap_np):
    # ... (overlay_heatmap 함수 내용 복사) ...
    original_img_np = np.array(original_img_pil)

    original_h, original_w = original_img_np.shape[0], original_img_np.shape[1]
    heatmap_resized = cv2.resize(heatmap_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    overlayed_img_bgr = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored, 0.4, 0)

    overlayed_img_rgb = cv2.cvtColor(overlayed_img_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(overlayed_img_rgb)


# PIL Image를 Base64 문자열로 변환하는 헬퍼 함수 (웹 페이지 표시용)
def pil_image_to_base64(image_pil):
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- Flask 라우트 정의 ---

# 메인 페이지 라우트: 이미지 업로드 폼을 보여줌
@app.route('/')
def index():
    # 'templates' 폴더 안의 index.html 파일을 렌더링하여 반환
    return render_template('index.html')

# 예측 처리 라우트: 이미지 파일을 받아 예측 및 Grad-CAM 수행
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "모델 로드 실패. 애플리케이션 설정을 확인하세요.", 500

    if 'file' not in request.files:
        return "파일이 업로드되지 않았습니다.", 400

    file = request.files['file']

    if file.filename == '':
        return "선택된 파일이 없습니다.", 400

    if file:
        try:
            # 1. 원본 이미지 로드 (PIL Image 객체로)
            # file.stream에서 이미지 데이터를 읽어 PIL Image로 열고 RGB 포맷으로 변환
            img_pil = Image.open(file.stream).convert('RGB')
            print("업로드된 이미지 로드 완료.")

            # --- 원본 이미지를 Base64로 변환 ---
            original_image_base64 = pil_image_to_base64(img_pil)
            print("원본 이미지 Base64 변환 완료.")

            # 2. 모델 입력용 이미지 전처리 및 디바이스 이동
            image_tensor = inference_transform_model_input(img_pil).unsqueeze(0).to(DEVICE)
            print("이미지 전처리 및 디바이스 이동 완료.")

            # 3. 이미지 분류 수행
            with torch.no_grad():
                output = model(image_tensor)

            probs = F.softmax(output, dim=1)[0]
            predicted_index = torch.argmax(probs).item()
            original_predicted_class_name = class_names[predicted_index]
            predicted_korean_name = class_name_mapping.get(original_predicted_class_name, original_predicted_class_name)

            prediction_text = f"예측 클래스: {predicted_korean_name}"
            probability_text = "각 클래스 확률:<br>"
            for i, prob in enumerate(probs.cpu().numpy()):
                original_class_name = class_names[i]
                korean_class_name = class_name_mapping.get(original_class_name, original_class_name)
                probability_text += f"&nbsp;&nbsp;{korean_class_name}: {prob:.4f}<br>"


            # 4. Grad-CAM 생성
            target_layer_index = 18 # TODO: 사용 중인 VGG 모델에 맞게 정확한 인덱스로 수정
            try:
                target_layer = model.features[target_layer_index]

                if not isinstance(target_layer, torch.nn.Conv2d):
                     print(f"오류: 타겟 레이어 features[{target_layer_index}]는 Conv2d가 아닙니다. 인덱스를 확인하세요.")
                     heatmap = None
                else:
                     print(f"Grad-CAM 생성을 위한 타겟 레이어: {target_layer.__class__.__name__} (features[{target_layer_index}])")
                     heatmap = get_grad_cam(model, image_tensor, target_layer, target_category=predicted_index)
                     print("Grad-CAM 히트맵 생성 완료.")

            except IndexError:
                 print(f"오류: features[{target_layer_index}] - 잘못된 레이어 인덱스입니다. 모델 구조를 확인하세요.")
                 heatmap = None
            except Exception as e:
                 print(f"Grad-CAM 생성 중 예상치 못한 오류 발생: {e}")
                 heatmap = None


            # 5. 히트맵 시각화 (원본 이미지에 오버레이) 및 Base64 변환
            grad_cam_image_base64 = None
            if heatmap is not None:
                overlayed_image_pil = overlay_heatmap(img_pil, heatmap) # 원본 이미지 사용
                print("히트맵 오버레이 이미지 생성 완료.")
                grad_cam_image_base64 = pil_image_to_base64(overlayed_image_pil)
                print("오버레이 이미지 Base64 변환 완료.")
            else:
                print("Grad-CAM 시각화 이미지를 생성할 수 없습니다.")


            # 6. 결과 페이지 렌더링 및 반환
            return render_template('result.html',
                                   prediction_text=prediction_text,
                                   probability_text=probability_text,
                                   original_image=original_image_base64, # 원본 이미지 Base64 데이터 전달
                                   grad_cam_image=grad_cam_image_base64)

        except Exception as e:
            print(f"이미지 처리 또는 예측 중 오류 발생: {e}")
            return f"처리 중 오류 발생: {e}", 500

# 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)