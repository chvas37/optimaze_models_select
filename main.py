import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Загрузка данных
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

# Обработка данных
def process_data(data):
    # Преобразование столбцов типа object в числовые
    object_columns = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in object_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Избавление от нулевых значений и дубликатов
    data = data.dropna().drop_duplicates()
    return data

# Класс нейронной сети
class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Обучение нейронной сети
def train_pytorch_classifier(model, criterion, optimizer, train_loader):
    epoch_losses = []
    for epoch in range(100):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(train_loader))
    print("Нейронная сеть (PyTorch) обучена.")
    return epoch_losses

# Функция построения графика обучения
def plot_learning_curve(losses):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("learning_curve.png")
    plt.show()

def plot_confusion_matrix(confusion_mat, model_name, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mat, annot=True, fmt='g')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()
# Обучение модели
def train_model(model, X_train, y_train):
    if isinstance(model, tuple):
        model, criterion, optimizer = model
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train.values), torch.LongTensor(y_train.values)), batch_size=32, shuffle=True)
        losses = train_pytorch_classifier(model, criterion, optimizer, train_loader)
        return {"model": model, "losses": losses}
    else:
        model.fit(X_train, y_train)
        return {"model": model}

# Оценка модели


def evaluate_model(model, X_test, y_test, model_name, losses=None):
    if isinstance(model, NeuralNetworkClassifier):
        model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X_test.values)
            outputs = model(inputs)
            _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()
        accuracy = accuracy_score(y_test, y_pred)
        if losses is not None:
            plot_learning_curve(losses)
        save_results({'Accuracy': accuracy}, model_name)
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        save_results({'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}, model_name)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок:")
    print(confusion_mat)
    plot_confusion_matrix(confusion_mat, model_name, filename=f"confusion_matrix_{model_name}.png")



def save_results(metrics, model_name, filename="model_results.txt"):
    with open(filename, "a") as file:
        file.write(f"Results for {model_name}:\n")
        for metric_name, metric_value in metrics.items():
            file.write(f"{metric_name}: {metric_value}\n")
        file.write("\n")

# Выбор модели
def select_model(X, y):
    print("Доступные модели:")
    print("1. RandomForestClassifier")
    print("2. LogisticRegression")
    print("3. KNeighborsClassifier")
    print("4. DecisionTreeClassifier")
    print("5. Support Vector Machine (SVM)")
    print("6. Neural Network (MLP)")
    print("7. Neural Network Classifier (PyTorch)")

    model_mapping = {
        '1': RandomForestClassifier(),
        '2': LogisticRegression(),
        '3': KNeighborsClassifier(),
        '4': DecisionTreeClassifier(),
        '5': SVC(),
        '6': MLPClassifier(max_iter=1000),
        '7': create_pytorch_classifier(len(X.columns), 64, len(set(y)))
    }

    choice = input("Выберите модель (введите номер): ")
    return model_mapping.get(choice, None)

# Создание нейронной сети PyTorch
def create_pytorch_classifier(input_size, hidden_size, output_size):
    model = NeuralNetworkClassifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

# Главная функция
def main():
    parser = argparse.ArgumentParser(description='Менеджер обучения моделей машинного обучения')
    parser.add_argument('--data', type=str, help='Путь к файлу данных', required=False)
    parser.add_argument('--target_column', type=str, help='Имя столбца с целевой переменной', required=False)

    args = parser.parse_args()

    # Загрузка и обработка данных
    data_path = args.data if args.data else input("Введите путь к файлу данных: ")
    data = load_data(data_path)
    if data is None:
        return
    target_column = args.target_column if args.target_column else input("Введите имя столбца с целевой переменной: ")
    if target_column not in data.columns:
        print(f"Столбец {target_column} не найден.")
        return
    data = process_data(data)

    # Разделение данных
    y = data[target_column]
    X = data.drop(target_column, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Выбор и обучение модели
    model = select_model(X, y)
    if model is None:
        print("Неверный выбор модели.")
        return
    model_data = train_model(model, X_train, y_train)
    model = model_data["model"]
    losses = model_data.get("losses", None)

    # Оценка модели
    evaluate_model(model, X_test, y_test, model.__class__.__name__, losses)

if __name__ == '__main__':
    main()
