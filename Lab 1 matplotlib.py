import numpy as np
import matplotlib.pyplot as plt

# Задание 1
# Вариант Y(x) = (1/x) * sin(5x), x = [-5...5]
x = np.linspace(-5, 5, 1000)  # вычисление значений x в интервале

# Заменим значения x, равные 0, на небольшие значения, чтобы избежать деления на ноль
epsilon = 1e-10
x = np.where(x == 0, epsilon, x)

y = (1/x) * np.sin(5 * x)

plt.plot(x, y)
plt.title('График функции Y(x) = (1/x) * sin(5x)')
plt.grid(True)
plt.show()

# Задание 2
with open('text.txt', 'r') as f:
    content = f.read()

freq_dict = {}
for char in content:
    if char.isalpha():
        char = char.lower()
        freq_dict[char] = freq_dict.get(char, 0) + 1

# Построение гистограммы
letters = list(freq_dict.keys())
counts = list(freq_dict.values())

plt.bar(letters, counts, color='lightblue')
plt.xlabel('Буквы')
plt.ylabel('Частота')
plt.title('Гистограмма частоты появления букв в тексте')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Задание 3
with open('text.txt', 'r') as f:
    content = f.read()

dot_count = content.count('.') - content.count('...')
question_count = content.count('?')
exclaim_count = content.count('!')
ellipsis_count = len(re.findall(r'\.\.\.(?!\w)', content))

categories = ['Обычные', 'Вопросительные', 'Окличные', 'Триточка']
counts = [dot_count, question_count, exclaim_count, ellipsis_count]

plt.bar(categories, counts, color='lightcoral')
plt.xlabel('Типы предложений')
plt.ylabel('Частота')
plt.title('Гистограмма частоты появления типов предложений')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()