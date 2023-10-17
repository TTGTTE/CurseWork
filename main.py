import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping

# Функция для загрузки данных и извлечения аудиопризнаков
def load_data(folder):
    X, y = [], []       # Инициализация массивов для признаков и меток
    labels = []         # Список для хранения меток

    # Обход всех подкаталогов, соответствующих классам
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):

            # Обход всех аудиофайлов внутри каждого класса
            for filename in os.listdir(class_folder):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(class_folder, filename)
                    
                    # Загрузка аудиофайла с частотой дискретизации 44100 и длительностью 3 секунды
                    audio, _ = librosa.load(audio_path, sr=44100, duration=3.0)

                    # Извлечение MFCC (Mel-frequency cepstral coefficients) и Chroma признаков
                    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
                    chroma = librosa.feature.chroma_stft(y=audio, sr=44100)

                    # Объединение MFCC и Chroma в один массив признаков
                    features = np.vstack((mfccs, chroma))
                    features = np.expand_dims(features, axis=-1)  # Расширение размерности для соответствия формату входных данных модели
                    X.append(features)
                    labels.append(class_name)

    # Преобразование меток в бинарный формат
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([[label] for label in labels])

    # Сохранение классов для последующего использования
    np.save('classes.npy', mlb.classes_)
    return np.array(X), np.array(y)

# Функция для создания модели нейронной сети
def create_model(input_shape, num_labels):
    # Инициализация модели
    model = tf.keras.Sequential([

        # Два последовательных LSTM слоя
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(input_shape[0], input_shape[1])),
        tf.keras.layers.LSTM(128),

        # Полносвязные слои
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Слой Dropout для регуляризации
        tf.keras.layers.Dense(128, activation='relu'),

        # Выходной слой с сигмоидной активацией для многоклассовой классификации
        tf.keras.layers.Dense(num_labels, activation='sigmoid')
    ])
    # Компиляция модели с оптимизатором Adam и функцией потерь binary_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Функция для классификации аудиофайла
def classify_audio(model, mlb, audio_path):

    # Загрузка аудиофайла и извлечение признаков
    audio, _ = librosa.load(audio_path, sr=44100, duration=3.0)
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=44100)
    features = np.vstack((mfccs, chroma))
    features = np.expand_dims(features, axis=[0, -1])

    # Прогнозирование с использованием модели
    predictions = model.predict(features)

    # Пороговое значение для классификации
    thresholded_output = (predictions >= 0.5).astype(np.int64)

    # Инвертирование бинарных меток в имена классов
    predicted_labels = mlb.inverse_transform(thresholded_output)[0]
    return predicted_labels

# Главная функция для управления всем процессом
def main():
    model = None                    # Инициализация модели
    mlb = MultiLabelBinarizer()     # Инициализация бинаризатора меток

    # Загрузка сохраненных классов, если они существуют
    if os.path.exists('classes.npy'):
        mlb.classes_ = np.load('classes.npy', allow_pickle=True)

    # Инициализация ранней остановки для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    while True:
        print("Выберите действие:")
        print("1. Обучить нейронную сеть")
        print("2. Протестировать нейронную сеть")
        print("3. Экспорт обученной нейронной сети")
        print("4. Импорт обученной нейронной сети")
        print("5. Классифицировать аудиофайл")
        choice = input("> ")

        # Обучение модели
        if choice == '1':
            X, y = load_data("Dataset")                     # Загрузка данных
            X_val, y_val = load_data("Validation")          # Загрузка валидационных данных
            model = create_model(X[0].shape, y.shape[1])    # Создание модели

            # Обучение модели с ранней остановкой и валидационным набором данных
            model.fit(X, y, epochs=1000, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Тестирование модели
        elif choice == '2':
            X_test, y_test = load_data("Testing")               # Загрузка тестовых данных
            loss, accuracy = model.evaluate(X_test, y_test)     # Оценка модели
            print(f"Тестовая потеря: {loss}, Тестовая точность: {accuracy}")

        # Экспорт модели
        elif choice == '3':
            model.save("trained_model.h5")  # Сохранение модели

        # Импорт модели
        elif choice == '4':
            model = tf.keras.models.load_model("trained_model.h5")  # Загрузка модели

        # Классификация аудиофайла
        elif choice == '5':
            audio_path = input("Введите путь к аудиофайлу: ")
            labels = classify_audio(model, mlb, audio_path)  # Получение меток
            print(f"Предсказанные метки: {labels}")

        else:
            print("Неизвестный выбор. Пожалуйста, попробуйте еще раз.")

if __name__ == "__main__":
    main()
