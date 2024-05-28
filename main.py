import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 데이터셋 불러오기
data = pd.read_csv('db.csv')  # 'your_dataset.csv'에는 데이터셋 파일명이 들어갑니다.
data = data.dropna()
# 입력 및 출력 변수 설정
X = data[['Year', 'Mileage', 'Condition_New', 'Condition_Used', 'Make_Acura', 'Make_Alfa', 'Make_Aston', 'Make_Audi', 'Make_BMW', 'Make_Bentley', 'Make_Bugatti', 'Make_Buick', 'Make_Cadillac', 'Make_Chevrolet', 'Make_Chrysler', 'Make_Delorean', 'Make_Dodge', 'Make_FIAT', 'Make_Ferrari', 'Make_Fisker', 'Make_Ford', 'Make_GMC', 'Make_Genesis', 'Make_Honda', 'Make_Hummer', 'Make_Hyundai', 'Make_INFINITI', 'Make_Isuzu', 'Make_Jaguar', 'Make_Jeep', 'Make_Kia', 'Make_Lamborghini', 'Make_Land', 'Make_Lexus', 'Make_Lincoln', 'Make_Lotus', 'Make_MINI', 'Make_Maserati', 'Make_Mazda', 'Make_McLaren', 'Make_Mercedes-Benz', 'Make_Mercury', 'Make_Mitsubishi', 'Make_Nissan', 'Make_Oldsmobile', 'Make_Plymouth', 'Make_Polestar', 'Make_Pontiac', 'Make_Porsche', 'Make_RAM', 'Make_Rivian', 'Make_Rolls-Royce', 'Make_Saab', 'Make_Saturn', 'Make_Scion', 'Make_Subaru', 'Make_Tesla', 'Make_Toyota', 'Make_Triumph', 'Make_Volkswagen', 'Make_Volvo']]
y = data['Price']

# 데이터 표준화 (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 및 테스트 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential()
model.add(Dense(61, input_dim=61, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))  # 출력층에 활성화 함수 설정하지 않음

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가 (평균 제곱 오차)
mse = tf.keras.losses.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse.numpy())

# 새로운 데이터에 대한 예측 예시
new_data = [[2023, 20000, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
new_data_scaled = scaler.transform(new_data)

#BMW, 32000km, 중고차 => [59142.637] 가격
# Mean Squared Error: [3.0867402e+09 3.3652493e+09 2.9051090e+09 ... 2.9165604e+09 2.9034194e+09]

predicted_price = model.predict(new_data_scaled)
print("Predicted Price for the new car:", predicted_price)