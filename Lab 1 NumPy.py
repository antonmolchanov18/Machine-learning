import numpy as np

# Задание 1
print('Решение первой задачи:')

arr1 = np.array([1, 2, 3, 4, 5, 6])
arr2 = np.array([7, 8, 9, 10, 11, 12])

# Выполнение арифметических операций
sum_result = np.add(arr1, arr2)
print(f'Результат сложения: {sum_result}')

diff_result = np.subtract(arr1, arr2)
print(f'Результат вычитания: {diff_result}')

prod_result = np.multiply(arr1, arr2)
print(f'Результат умножения: {prod_result}')

quot_result = np.divide(arr1, arr2)
print(f'Результат деления: {quot_result}')

# Объединение массивов
merged_array = np.concatenate((arr1, arr2))
print(f'Объединённый массив: {merged_array}')

# Наибольший и наименьший элементы, сумма и произведение всех элементов
max_val = np.amax(merged_array)
print(f'Максимальное значение: {max_val}')

min_val = np.amin(merged_array)
print(f'Минимальное значение: {min_val}')

total_sum = np.sum(merged_array)
print(f'Общая сумма элементов: {total_sum}')

total_prod = np.prod(merged_array)
print(f'Общее произведение элементов: {total_prod}')

# Задание 2
print('Решение второй задачи:')

arr1 = np.array([5, 4, 1, 67, 32, 6, 16, 11, 9, 10, 9, 3, 13, 2, 15])

# Подсчёт среднего значения и коррекция каждого элемента
mean_val = np.mean(arr1)
corrected_arr = arr1 - mean_val
print(f'Корректированный массив: {corrected_arr}')

# Сортировка массива по возрастанию
sorted_arr = np.sort(corrected_arr)
print(f'Отсортированный массив: {sorted_arr}')

# Задание 3
print('Решение третьей задачи:')

# Создание одномерного массива из 20 элементов с использованием функции random()
random_arr = np.random.rand(20)
print(f'Одномерный массив: {random_arr}')

# Преобразование одномерного массива в двумерный массив размером 4x5
reshaped_arr = random_arr.reshape(4, 5)

# Увеличение каждого элемента двумерного массива на 10
increased_arr = reshaped_arr + 10
print(f'Двумерный массив после увеличения на 10:\n{increased_arr}')

# Задание 4
print('Решение четвертой задачи:')

# Создание двумерного массива целых чисел в диапазоне от -15 до 15 размером 5x5
array2d = np.random.randint(-15, 16, (5, 5))
print(f'Изначальный массив:\n{array2d}')

# Замена отрицательных чисел на -1 и положительных на 1
array2d[array2d < 0] = -1
array2d[array2d > 0] = 1
print(f'Массив после замены значений:\n{array2d}')

# Задание 5
print('Решение пятой задачи:')

# Инициализация матриц A и B
matrix_A = np.array([[2, 3, -1], [4, 5, 2], [-1, 0, 7]])
matrix_B = np.array([[-1, 0, 5], [0, 1, 3], [2, -2, 4]])

# Выполнение операций с матрицами
operation_result = 2 * (matrix_A + matrix_B) * (2 * matrix_B - matrix_A)
print(f'Результат операции:\n{operation_result}')

# Задание 6
print('Решение шестой задачи:')

# Матрица коэффициентов для системы линейных уравнений
coeff_matrix = np.array([[1, 1, 2, 3],
                         [3, -1, -2, -2],
                         [2, -3, -1, -1],
                         [1, 2, 3, -1]])

rhs = np.array([1, -4, -6, -4])  # Вектор правой части уравнений

# Решение системы линейных уравнений
sol = np.linalg.solve(coeff_matrix, rhs)
print(f'Решение системы уравнений: {sol}')
