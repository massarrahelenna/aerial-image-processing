import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Variáveis globais
rect_start = None
rect_end = None
drawing = False
samples = []
labels = []
image = None
original_image = None
current_label = None
collected = 0
target_per_class = 0
zoom_level = 1.0
zoom_center = (0, 0)

# Desenha retângulo e captura amostras
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing, samples, labels, collected, target_per_class, image, zoom_level, zoom_center

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start = get_original_coords(x, y)
        rect_end = rect_start

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect_end = get_original_coords(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = get_original_coords(x, y)
        x1, y1 = rect_start
        x2, y2 = rect_end

        if x1 != x2 and y1 != y2:
            sample = original_image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            if sample.size != 0:
                h, w = sample.shape[:2]
                if h < 10 or w < 10:
                    print("Amostra muito pequena. Ignorada.")
                    return
                samples.append(sample)
                labels.append(current_label)
                collected += 1
                print(f"Amostra coletada para classe '{current_label}' ({collected}/{target_per_class})")

    # Zoom com roda do mouse
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            change_zoom(1.2, x, y)
        else:
            change_zoom(1/1.2, x, y)

# Mapeia coordenadas da tela para imagem original
def get_original_coords(x, y):
    global zoom_level, zoom_center
    zx, zy = zoom_center
    h, w = original_image.shape[:2]
    new_w, new_h = int(w / zoom_level), int(h / zoom_level)

    left = max(zx - new_w // 2, 0)
    top = max(zy - new_h // 2, 0)

    x_original = int(left + x / zoom_level)
    y_original = int(top + y / zoom_level)

    x_original = min(w - 1, max(0, x_original))
    y_original = min(h - 1, max(0, y_original))

    return (x_original, y_original)

# Atualiza zoom
def change_zoom(factor, mouse_x, mouse_y):
    global zoom_level, zoom_center
    zoom_level *= factor
    zoom_level = min(max(zoom_level, 1.0), 5.0)

    zoom_center = get_original_coords(mouse_x, mouse_y)

# Mostra imagem com zoom
def get_zoomed_image():
    global zoom_level, zoom_center
    h, w = original_image.shape[:2]
    new_w, new_h = int(w / zoom_level), int(h / zoom_level)

    cx, cy = zoom_center
    left = max(cx - new_w // 2, 0)
    right = min(left + new_w, w)
    top = max(cy - new_h // 2, 0)
    bottom = min(top + new_h, h)

    cropped = original_image[top:bottom, left:right]
    return cv2.resize(cropped, (w, h))

# Normalização
def normalize_samples(samples):
    norm_samples = []
    for sample in samples:
        sample_resized = cv2.resize(sample, (64, 64))
        norm_samples.append(sample_resized.flatten() / 255.0)
    return np.array(norm_samples)

def main():
    global image, original_image, current_label, collected, target_per_class, zoom_center

    image_path = os.path.join('/home', 'massarrahelenna', 'ProcessamentoInteligente', 'ProcessamentoInteligente', 'Aula-Pratica', 'Imagens', '4.jpg')
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return

    zoom_center = (original_image.shape[1] // 2, original_image.shape[0] // 2)

    num_classes = int(input("Digite o número de classes: "))
    target_per_class = int(input("Digite o número de amostras por classe: "))
    classes = [input(f"Digite o nome da classe {i+1}: ") for i in range(num_classes)]

    cv2.namedWindow("Imagem")
    cv2.setMouseCallback("Imagem", draw_rectangle)

    for current_label in classes:
        collected = 0
        print(f"\nColetando amostras para a classe '{current_label}'.")
        print("Use o mouse para selecionar e a roda do mouse para dar zoom.")

        while collected < target_per_class:
            image = get_zoomed_image()
            img_copy = image.copy()
            if rect_start and rect_end:
                pt1 = (int((rect_start[0] - zoom_center[0]) * zoom_level + image.shape[1] // 2),
                       int((rect_start[1] - zoom_center[1]) * zoom_level + image.shape[0] // 2))
                pt2 = (int((rect_end[0] - zoom_center[0]) * zoom_level + image.shape[1] // 2),
                       int((rect_end[1] - zoom_center[1]) * zoom_level + image.shape[0] // 2))
                cv2.rectangle(img_copy, pt1, pt2, (0, 255, 0), 2)

            cv2.imshow("Imagem", img_copy)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                print("Interrupção pelo usuário.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    if len(samples) == 0:
        print("Nenhuma amostra coletada.")
        return

    X = normalize_samples(samples)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\nTotal de amostras coletadas: {len(samples)}")
    for label in classes:
        print(f"{label}: {labels.count(label)} amostras")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(X_train) < 1:
        print("Número insuficiente de amostras para treinar.")
        return

    n_neighbors = min(3, len(X_train))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)

    print(f"\nAcurácia do modelo KNN: {acc:.2f}")

    for i in range(min(5, len(X_test))):
        pred = knn.predict([X_test[i]])[0]
        real = y_test[i]
        print(f"Exemplo {i+1}: Real = {le.inverse_transform([real])[0]}, Previsto = {le.inverse_transform([pred])[0]}")

    print("\nMostrando exemplos classificados...")
    for i in range(min(5, len(X_test))):
        sample = (X_test[i] * 255).reshape(64, 64, 3).astype(np.uint8)
        label_pred = le.inverse_transform([knn.predict([X_test[i]])[0]])[0]
        cv2.putText(sample, f"{label_pred}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow(f"Exemplo {i+1}", sample)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    np.savez("dados_coletados.npz", samples=np.array(samples, dtype=object), labels=np.array(labels))
    print("Dados salvos em 'dados_coletados.npz'.")

if __name__ == "__main__":
    main()
