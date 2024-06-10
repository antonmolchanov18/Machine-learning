import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

#Варіант 2
# Задаємо діапазон x і обчислюємо y за новою функцією
x = np.arange(0, 10, 0.1)
y = x**3 - 10*x**2 + x

# Розділяємо дані на навчальні (тільки пари з парними індексами)
x_train = x[::2].reshape(-1, 1)
y_train = y[::2]

# Створюємо поліноміальні ознаки 13 ступеня
polynom = PolynomialFeatures(degree=13)

# Лінійна регресія без регуляризації
model = make_pipeline(polynom, LinearRegression())
model.fit(x_train, y_train)
y_pred = model.predict(x.reshape(-1, 1))

# Візуалізація результатів
plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Навчальні дані', color='red')
plt.plot(x, y, label='Первісна функція', color='green')
plt.plot(x, y_pred, label='Поліноміальна регресія (13 ступінь)', color='blue')
plt.legend()
plt.title('Поліноміальна регресія без регуляризації')
plt.show()

# Обчислення помилки
mse = mean_squared_error(y, y_pred)
print(f'Середньоквадратична помилка (без регуляризації): {mse}')

# Ridge регресія (L2 регуляризація)
model_ridge = make_pipeline(polynom, Ridge(alpha=1.0))
model_ridge.fit(x_train, y_train)
y_pred_ridge = model_ridge.predict(x.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Навчальні дані', color='red')
plt.plot(x, y, label='Первісна функція', color='green')
plt.plot(x, y_pred_ridge, label='Ridge регресія (L2)', color='blue')
plt.legend()
plt.title('Поліноміальна регресія з L2 регуляризацією')
plt.show()

# Обчислення помилки для Ridge регресії
mse_ridge = mean_squared_error(y, y_pred_ridge)
print(f'Середньоквадратична помилка з L2 регуляризацією: {mse_ridge}')

# Lasso регресія (L1 регуляризація)
model_lasso = make_pipeline(polynom, Lasso(alpha=1e-3, max_iter=10000))
model_lasso.fit(x_train, y_train)
y_pred_lasso = model_lasso.predict(x.reshape(-1, 1))

plt.figure(figsize=(12, 12))
plt.scatter(x_train, y_train, label='Навчальні дані', color='red')
plt.plot(x, y, label='Первісна функція', color='green')
plt.plot(x, y_pred_lasso, label='Lasso регресія (L1)', color='blue')
plt.legend()
plt.title('Поліноміальна регресія з L1 регуляризацією')
plt.show()

# Обчислення помилки для Lasso регресії
mse_lasso = mean_squared_error(y, y_pred_lasso)
print(f'Середньоквадратична помилка з L1 регуляризацією: {mse_lasso}\n')

# Порівняння всіх моделей
print(f'Середньоквадратична помилка (без регуляризації): {mse}')
print(f'Середньоквадратична помилка з L2 регуляризацією: {mse_ridge}')
print(f'Середньоквадратична помилка з L1 регуляризацією: {mse_lasso}')
