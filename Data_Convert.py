import pandas as pd


#Make,Year,Condition,Mileage,Price
# CSV 파일을 읽어옴
# df = pd.read_csv('./data/data_set.csv', delimiter=",")

# Make의 종류
# make_types = df['Make'].unique()
# Condition의 종류
# condition_types = df['Condition'].unique()

# print("Make의 종류:", make_types)
# print("Condition의 종류:", condition_types)



#-----------------------------------------------------------------
# Make와 Condition 열에 대해 원핫 인코딩 수행
# make_one_hot = pd.get_dummies(df['Make'], prefix='Make')
# condition_one_hot = pd.get_dummies(df['Condition'], prefix='Condition')

# # 원핫 인코딩된 열을 원래 데이터프레임에 추가
# df = pd.concat([df, make_one_hot, condition_one_hot], axis=1)

# # 원래의 Make와 Condition 열을 삭제
# df = df.drop(['Make', 'Condition'], axis=1)

# # 새로운 CSV 파일로 저장
# df.to_csv('./data/data_one_hot_encoded.csv', index=False)

# print("원핫 인코딩된 데이터가 'data_one_hot_encoded.csv' 파일로 저장되었습니다.")
#-------------------------------------------------------------------

df = pd.read_csv('./data/data_one_hot_encoded.csv')

# 데이터 값 변환: False는 0으로, True는 1로
df = df.replace({False: 0, True: 1})

# 새로운 CSV 파일로 저장
df.to_csv('./data/data_one_hot_encoded.csv', index=False)
