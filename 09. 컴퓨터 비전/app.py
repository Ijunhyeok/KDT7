import streamlit as st
import joblib
from PIL import Image
import numpy as np
import pandas as pd
import utils
import cv2
import streamlit as st

import random

color_content_map = {
    '사랑(라이트 핑크)': {
        'emotion': '부드러움, 사랑스러움, 순수함',
        'places': [
            "제주 카멜리아 힐", "네덜란드 큐켄호프 정원", "서울 올림픽공원 장미광장",
            "파리의 사랑의 다리", "경주 보문단지 핑크 정원"
        ],
        'quotes': [
            "따뜻한 마음은 라이트 핑크처럼 부드럽게 퍼져요.",
            "사랑스러운 하루, 당신과 어울리는 시간이에요.",
            "순수한 감정이 오늘을 채우는 순간.",
            "부드러움 속에 담긴 강인함을 아는 당신."
        ],
        'music': [
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",  # Perfect - Ed Sheeran
            "https://www.youtube.com/watch?v=LjhCEhWiKXk"   # Just the Way You Are - Bruno Mars
        ]
    },
    '행복(핫 핑크)': {
        'emotion': '열정, 활기, 강렬함',
        'places': [
            "미국 뉴욕 타임스퀘어", "서울 성수동 핫핑크 카페 거리", "라스베이거스 네온거리",
            "일본 도쿄 시부야"
        ],
        'quotes': [
            "강렬한 핑크처럼 당신은 세상을 빛내요.",
            "열정으로 가득 찬 하루, 당신답습니다.",
            "활기 넘치는 순간, 당신과 함께."
        ],
        'music': [
            "https://www.youtube.com/watch?v=nfWlot6h_JM",  # Shake It Off - Taylor Swift
            "https://www.youtube.com/watch?v=OPf0YbXqDm0"   # Uptown Funk - Bruno Mars
        ]
    },
    '희망(딥 핑크)': {
        'emotion': '자신감, 매력, 우아함',
        'places': [
            "프랑스 니스의 해변", "제주 함덕해수욕장 석양", "스페인 바르셀로나 가우디의 거리"
        ],
        'quotes': [
            "딥 핑크처럼 매혹적인 순간이 당신을 기다려요.",
            "당신의 자신감은 세상의 중심이에요.",
            "우아함과 매력이 가득 찬 당신의 하루."
        ],
        'music': [
            "https://www.youtube.com/watch?v=rYEDA3JcQqw"  # Rolling in the Deep - Adele
        ]
    },
    '우아함(페일 핑크)': {
        'emotion': '차분함, 고요함, 은은함',
        'places': [
            "프랑스 프로방스 라벤더 밭", "스위스 인터라켄 호숫가", "서울 남산 둘레길"
        ],
        'quotes': [
            "은은한 페일 핑크처럼, 차분한 당신이 참 아름다워요.",
            "고요함 속에서 피어나는 빛깔, 당신만의 색이에요.",
            "평온함을 느끼는 순간이 당신에게 가득하길."
        ],
        'music': [
            "https://www.youtube.com/watch?v=CvFH_6DNRCY"  # Clair de Lune - Debussy
        ]
    },
    '편안함(더스티 로즈)': {
        'emotion': '깊이, 따뜻함, 차분함',
        'places': [
            "캐나다 밴프 국립공원", "제주 숲길 힐링 코스", "영국 레이크 디스트릭트"
        ],
        'quotes': [
            "로즈우드 핑크처럼 깊고 따뜻한 마음을 가진 당신.",
            "조용한 숲속의 평화로움이 당신의 하루를 감싸요.",
            "차분한 아름다움은 당신에게 어울리는 색입니다."
        ],
        'music': [
            "https://www.youtube.com/watch?v=RgKAFK5djSk"  # See You Again - Wiz Khalifa
        ]
    },
    '순수함(라이트 코랄)': {
        'emotion': '상쾌함, 희망, 활기',
        'places': [
            "경남 하동 십리벚꽃길", "하와이 와이키키 해변", "이탈리아 아말피 코스트"
        ],
        'quotes': [
            "상쾌하고 밝은 에너지로 가득한 당신.",
            "희망과 활기가 넘치는 라이트 코랄처럼!",
            "빛나는 하루를 선물합니다."
        ],
        'music': [
            "https://www.youtube.com/watch?v=ZbZSe6N_BXs"  # Happy - Pharrell Williams
        ]
    },
    '기타(슬레이트 그레이)': {
        'emotion': '신뢰, 안정감, 차분함',
        'places': [
            "일본 교토 고즈넉한 절벽길", "그리스 산토리니 바닷가", "강릉 안목 커피 거리"
        ],
        'quotes': [
            "슬레이트 그레이처럼 묵직한 안정감을 가진 당신.",
            "차분한 힘은 당신의 고유한 매력입니다.",
            "바람처럼 잔잔한 마음이 당신 곁에 머물길."
        ],
        'music': [
            "https://www.youtube.com/watch?v=hLQl3WQQoQ0",  # Someone Like You - Adele
            "https://www.youtube.com/watch?v=ttEMYvpoR-k"   # Hallelujah - Leonard Cohen
        ]
    }
}

# print("🎨 색상별 감정과 추천 콘텐츠가 준비되었습니다!")

# Streamlit 시작
st.title("""봄철 분홍계열 꽃의 색상에 관해 추천드려요.""")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_container_width=True)

    # 이미지 저장 후 전처리
    temp_path = "temp_uploaded.jpg"
    image.save(temp_path)

    # 모델 로드
    bundle = joblib.load('decision_tree_model.pkl')
    dt_model = bundle['model']
    features = bundle['features']

    # 특징 추출
    df_input = utils.extract_avg_std_features(temp_path)
    df_input = df_input[features]

    # 예측
    pred = dt_model.predict(df_input)[0]
    # proba = dt_model.predict_proba(df_input)[0]
    # class_names = dt_model.classes_

    # st.subheader(f"예측된 Color: {pred}")
    # st.write("예측 확률 분포:")
    # st.bar_chart(pd.DataFrame(proba, index=class_names, columns=['확률']))

    # 예측 결과를 기반으로 콘텐츠 선택
    color_group = pred

    content = color_content_map.get(color_group, {
        'emotion': '매력',
        'places': ['제주 함덕해수욕장 석양'],
        'quotes': ['강렬한 핑크처럼 당신은 세상을 빛내요.'],
        'music': ['https://www.youtube.com/watch?v=rYEDA3JcQqw']
    })

    place = random.choice(content['places'])
    quote = random.choice(content['quotes'])
    music_url = random.choice(content['music'])
    emotion = content['emotion']

    st.markdown(f"**어울리는 장소:** {place}")
    st.markdown(f"**색상이 주는 감정:** {emotion}")
    st.markdown(f"**감성 글귀:** _{quote}_")

 
    if music_url:
        st.markdown(f"[YouTube에서 감성 음악 듣기]({music_url})") 
    else:
        st.info("어울리는 음악이 아직 준비되지 않았어요!")
