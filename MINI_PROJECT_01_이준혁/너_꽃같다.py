import random
import pandas as pd

# 엑셀 파일 읽기
df = pd.read_excel('./꽃_이름,꽃말.xlsx', engine='openpyxl') # 엑셀 파일 경로를 지정하세요.
# df = pd.read_excel('path_to_your_excel_file.xlsx', engine='openpyxl') # 엑셀 파일 경로를 지정하세요.

# print(df.columns)
# print(df.shape)

# 딕셔너리로 변환
flower_dict = {}
for index, row in df.iterrows():
    flower_name = row['꽃 이름'] # 엑셀의 열 이름에 맞게 수정
    meanings = tuple(row['꽃말'].split(',')) # 쉼표로 구분된 꽃말을 튜플로 변환
    flower_dict[flower_name] = meanings

all_meanings = []
for meanings in flower_dict.values():
    all_meanings.extend(meanings)

# 랜덤하게 6개 선택하여 튜플로 저장
random_flo_mean = tuple(random.sample(all_meanings, 6))

while True:
    print(f'\n{random_flo_mean}')
    want_flower = input("\n다음 꽃말 중 어느 꽃말이 궁금하신가요? ").strip()
    if want_flower in random_flo_mean: # 너가 찾는 꽃말이 랜덤꽃말에 있니?
        matching_flowers = [] # 매칭한 값 저장할 빈 리스트
        selected_history = []  # 선택한 기록 저장용 빈 리스트
        
        for flower, meanings in flower_dict.items(): # 꽃 딕셔너리에서 한줄씩 key, value를 가져와라
            if want_flower in meanings: # 꽃말과 같은 입력값이 있다면
                matching_flowers.append(flower) # 저장소에 추가하라
        # 값을 입력하면 꽃 딕셔너리를 한줄씩 반복 검증하며 해당 값을 가진 key를 따로 저장함.
        
        selected_history.append((want_flower, matching_flowers)) # 앞서 검증된 값과 키를 저장
        
        if len(matching_flowers) > 1:
            print(f"\n'{want_flower}'의 꽃말을 지닌 꽃은 여러개 있습니다.")
            new_choices = {} # 새롭게 담을 빈 주소
            for flower in matching_flowers: # 매칭된 꽃에 기존 꽃이 있니?
                other_meanings = set(flower_dict[flower]) - {want_flower} # 있으면 기존 꽃 딕셔너리의 꽃 이름과 해당하는 리스트에서 입력 받은 꽃말을 제외해서 따로 저장해라
                new_choices[flower] = random.choice(list(other_meanings)) # 입력 받은 꽃말을 제외한 리스트를 다시 한번 랜덤으로 고른후 새롭게 담을 주소에 리스트로 저장
            
            available_meanings = tuple(new_choices.values()) # 꽃말 튜플로 저장 및 출력
            print(f"\n한번더 다음 꽃말 중 하나를 선택해주세요: {available_meanings}")
            
            while True:
                new_choice = input().strip()
                if new_choice in available_meanings:
                    for flower, meaning in new_choices.items():
                        if new_choice == meaning:
                            # 선택한 꽃말과 꽃 저장
                            selected_history.append((new_choice, flower))
                            
                            # 2번 이상 선택했을 경우 기록 출력
                            if len(selected_history) >= 2:
                                chosen_meanings = [meaning for meaning, other_mean in selected_history]
                                final_flower = selected_history[-1][1]
                                print(f"\n지금까지 선택한 꽃말은 {chosen_meanings}입니다. 해당 꽃말들을 '{final_flower}'의 꽃말입니다.")
                            break
                    break
                else:
                    print(f"제시된 꽃말 {available_meanings} 중에서 선택해주세요.")
        else: # 하나만 있는 값의 꽃말이면 바로 이렇게 출력해라
            print(f"\n'{want_flower}'을/를 선택하셨네요. 해당 꽃말은 '{matching_flowers}'의 꽃말입니다.")
        break
    else: # 입력한 꽃말이 기존 value의 값과 다를때 출력
        print("\n앞서 제시된 꽃말 중에서 입력해 주십시오.")