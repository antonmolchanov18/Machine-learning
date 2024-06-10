import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('coffee_ratings.csv')

# Выбор необходимых столбцов и удаление пропущенных значений
columns = ['total_cup_points', 'species', 'country_of_origin', 'variety', 'aroma', 'aftertaste',
           'acidity', 'body', 'balance', 'sweetness', 'altitude_mean_meters', 'moisture']
data = data[columns].dropna()

# Кодирование категориальных переменных
ord_enc = OrdinalEncoder()
for col in ['species', 'country_of_origin', 'variety']:
    data[col] = ord_enc.fit_transform(data[[col]])

# Разделение на признаки и целевую переменную
X = data.drop('total_cup_points', axis=1)
y = data['total_cup_points']

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Функция для обучения и оценки модели
def train_evaluate(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1, classification_report(y_test, y_pred)

def plot_results(x_values, accuracy_results, f1_results, x_label):
    plt.plot(x_values, accuracy_results, marker='o', label='Accuracy')
    plt.plot(x_values, f1_results, marker='o', label='F1-score')
    plt.xlabel(x_label)
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# Исследование количества деревьев в лесу
n_estim_list = [10, 50, 100, 200, 500]
accuracy_results = []
f1_results = []

for n_est in n_estim_list:
    accuracy, f1, _ = train_evaluate(X_train, X_test, y_train, y_test, n_estimators=n_est)
    accuracy_results.append(accuracy)
    f1_results.append(f1)
    print(f'Accuracy with {n_est} trees: {accuracy}')
    print(f'F1-score with {n_est} trees: {f1}')

plot_results(n_estim_list, accuracy_results, f1_results, 'Number of Trees')

# Исследование максимальной глубины деревьев
max_depth_list = [None, 10, 20, 50, 100]
accuracy_results = []
f1_results = []

for max_depth in max_depth_list:
    accuracy, f1, _ = train_evaluate(X_train, X_test, y_train, y_test, max_depth=max_depth)
    accuracy_results.append(accuracy)
    f1_results.append(f1)
    print(f'Accuracy max depth {max_depth}: {accuracy}')
    print(f'F1-score max depth {max_depth}: {f1}')

plot_results(max_depth_list, accuracy_results, f1_results, 'Max Depth')

# Исследование минимального количества образцов для разделения узла
min_samples_split_list = [2, 5, 10, 20, 50]
accuracy_results = []
f1_results = []

for min_samples in min_samples_split_list:
    accuracy, f1, _ = train_evaluate(X_train, X_test, y_train, y_test, min_samples_split=min_samples)
    accuracy_results.append(accuracy)
    f1_results.append(f1)
    print(f'Accuracy min samples split {min_samples}: {accuracy}')
    print(f'F1-score min samples split {min_samples}: {f1}')

plot_results(min_samples_split_list, accuracy_results, f1_results, 'Min Samples Split')

# Исследование минимального количества образцов для каждого листового узла
min_samples_leaf_list = [1, 2, 5, 10, 20]
accuracy_results = []
f1_results = []

for min_leaf in min_samples_leaf_list:
    accuracy, f1, _ = train_evaluate(X_train, X_test, y_train, y_test, min_samples_leaf=min_leaf)
    accuracy_results.append(accuracy)
    f1_results.append(f1)
    print(f'Accuracy min samples leaf {min_leaf}: {accuracy}')
    print(f'F1-score min samples leaf {min_leaf}: {f1}')

plot_results(min_samples_leaf_list, accuracy_results, f1_results, 'Min Samples Leaf')
