# train_genre_model.py (SGDClassifier - 초고속 SVM 버전)

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. 데이터 로드 및 머지
DATA_DIR = './data'
books       = pd.read_csv(os.path.join(DATA_DIR, 'books_db.csv'))
book_genres = pd.read_csv(os.path.join(DATA_DIR, 'book_genres_db.csv'))
main_genres = pd.read_csv(os.path.join(DATA_DIR, 'main_genres_db.csv'))

df = (
    pd.merge(book_genres, books[['book_id','title']], on='book_id')
      .rename(columns={'title':'book_title'})
)
df = (
    pd.merge(df, main_genres[['main_genre_id','title']], on='main_genre_id')
      .rename(columns={'title':'main_genre_name'})
)
data = df[['book_title','main_genre_name']]

# 2. SBERT 임베딩
sbert = SentenceTransformer('all-MiniLM-L6-v2')
titles = data['book_title'].tolist()
X = sbert.encode(titles, show_progress_bar=True, batch_size=64)

# ✅ 장르 통합 매핑
genre_mapping = {
    'Crime, Thriller & Mystery': 'Fiction',
    'Fantasy, Horror & Science Fiction': 'Fiction',
    'Literature & Fiction': 'Fiction',
    'Teen & Young Adult': 'Fiction',
    'Children\'s Books': 'Children',
    'Comics & Mangas': 'Children',
    'Biographies, Diaries & True Accounts': 'Non-fiction',
    'History': 'Non-fiction',
    'Politics': 'Non-fiction',
    'Society & Social Sciences': 'Non-fiction',
    'Science & Mathematics': 'Science',
    'Medicine & Health Sciences': 'Science',
    'Sciences, Technology & Medicine': 'Science',
    'Computing, Internet & Digital Media': 'Science',
    'Engineering': 'Science',
    'Business & Economics': 'Business',
    'Exam Preparation': 'Education',
    'Higher Education Textbooks': 'Education',
    'School Books': 'Education',
    'Textbooks & Study Guides': 'Education',
    'Language, Linguistics & Writing': 'Education',
    'Crafts, Home & Lifestyle': 'Lifestyle',
    'Health, Family & Personal Development': 'Lifestyle',
    'Reference': 'Lifestyle',
    'Religion': 'Religion',
    'Law': 'Law',
    'Sports': 'Sports',
    'Travel': 'Travel',
    'Arts, Film & Photography': 'Arts'
}

# 3. 장르 통합 매핑 적용
data.loc[:, 'main_genre_name'] = data['main_genre_name'].map(genre_mapping).fillna(data['main_genre_name'])

# 3. 라벨 인코딩
le = LabelEncoder()
y = le.fit_transform(data['main_genre_name'])

# 4. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 모델 학습 (SGDClassifier - Linear SVM)
model = SGDClassifier(
    loss='hinge',         # Linear SVM
    penalty='l2',         # L2 정규화
    alpha=1e-4,           # 규제 강도
    max_iter=1000,        # 최대 반복수
    random_state=42,
    early_stopping=True,  # 조기 종료 활성화
    n_iter_no_change=5,   # 5번 동안 개선 없으면 종료
    validation_fraction=0.1, # 검증용 데이터 비율
    verbose=1             # 학습 로그 출력
)
model.fit(X_train, y_train)

# 6. 평가
y_pred = model.predict(X_test)

print('✅ Test Accuracy:', accuracy_score(y_test, y_pred))
print('\n✅ Classification Report:\n', classification_report(
    y_test, y_pred, target_names=le.classes_
))

# ✅ 저장할 폴더 준비
FIG_DIR = r'C:\Users\KDP-14\Desktop\KDT7\13_FLASK\MINI\PROJECT\figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ✅ 1. 혼동행렬(confusion matrix) 시각화 + 저장
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 저장
confusion_matrix_path = os.path.join(FIG_DIR, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path, bbox_inches='tight')
plt.close()
print(f'✅ Confusion matrix saved to: {confusion_matrix_path}')

# ✅ 2. confidence score(최대 예측 확률) 분포 시각화 + 저장
y_pred_prob = model.decision_function(X_test)
# decision_function은 margin을 반환하므로 softmax 비슷하게 변환하려면 np.max(abs(x), axis=1) 쓸 수 있어
confidence_scores = np.max(np.abs(y_pred_prob), axis=1)

plt.figure(figsize=(8,5))
sns.histplot(confidence_scores, bins=30, kde=True)
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence Score')
plt.ylabel('Number of Samples')

# 저장
confidence_distribution_path = os.path.join(FIG_DIR, 'confidence_distribution.png')
plt.savefig(confidence_distribution_path, bbox_inches='tight')
plt.close()
print(f'✅ Confidence distribution saved to: {confidence_distribution_path}')

# 8. 모델 저장
MODEL_DIR = r'C:\Users\KDP-14\Desktop\KDT7\13_FLASK\MINI\PROJECT\model'
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'genre_predictor_sgd.pkl')

with open(model_path, 'wb') as f:
    pickle.dump((model, le), f)

print(f'✅ Saved model to {model_path}')