import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ["variance", "skewness", "curtosis", "entropy", "class"]
data = pd.read_csv(url, names=columns, sep=",", header=None)

# Візуалізація даних
sns.pairplot(data, hue='class')
plt.show()

# Розділення на ознаки та цільову змінну
X = data.drop(columns='class')
y = data['class']

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання моделей SVM з різними ядрами
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svc_models = {kernel: SVC(kernel=kernel) for kernel in kernels}
predictions = {kernel: svc_models[kernel].fit(X_train, y_train).predict(X_test) for kernel in kernels}

# Функція для оцінки моделей
def evaluate_model(y_true, y_pred, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d').set_title(title)
    plt.show()
    print(classification_report(y_true, y_pred))

# Оцінка моделей з різними ядрами
for kernel in kernels:
    evaluate_model(y_test, predictions[kernel], f"{titles[kernel]} SVM")

# Дослідження впливу параметра C на модель з RBF ядром
C_values = [0.1, 1, 10, 100]
for C in C_values:
    svc_rbf = SVC(kernel='rbf', C=C)
    y_pred_C = svc_rbf.fit(X_train, y_train).predict(X_test)
    evaluate_model(y_test, y_pred_C, f'RBF SVM з C={C}')

# Дослідження впливу параметра gamma на модель з RBF ядром
gamma_values = [0.01, 0.1, 1, 10]
for gamma in gamma_values:
    svc_rbf = SVC(kernel='rbf', gamma=gamma)
    y_pred_gamma = svc_rbf.fit(X_train, y_train).predict(X_test)
    evaluate_model(y_test, y_pred_gamma, f'RBF SVM з gamma={gamma}')
