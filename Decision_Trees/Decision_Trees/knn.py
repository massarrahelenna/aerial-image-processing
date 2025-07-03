from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from tqdm import tqdm
import pandas as pd

# === 1. Carregar imagens ===
img_main_original = Image.open("BSB-1.jpg").convert("RGB")
img_main = img_main_original.resize((img_main_original.width // 2, img_main_original.height // 2))
img_road = Image.open("teste_1.png").convert("RGB")
img_vegetation = Image.open("teste_2.png").convert("RGB")
img_soil = Image.open("teste_3.png").convert("RGB")

# === 2. Extrair pixels e rótulos ===
def get_pixels_and_labels(image, label):
    pixels = np.array(image).reshape(-1, 3)
    labels = np.full((pixels.shape[0],), label)
    return pixels, labels

X_road, y_road = get_pixels_and_labels(img_road, 0)
X_veg, y_veg = get_pixels_and_labels(img_vegetation, 1)
X_soil, y_soil = get_pixels_and_labels(img_soil, 2)

X_train = np.vstack((X_road, X_veg, X_soil))
y_train = np.concatenate((y_road, y_veg, y_soil))

# === 3. Reduzir base para 10.000 amostras com balanceamento ===
X_train_sampled, y_train_sampled = resample(X_train, y_train, n_samples=10000, stratify=y_train, random_state=42)

# === 4. Treinar KNN com paralelismo ===
clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
clf.fit(X_train_sampled, y_train_sampled)

# === 5. Classificar imagem reduzida ignorando fundo preto ===
main_pixels = np.array(img_main)
height, width = img_main.size[1], img_main.size[0]
mask_valid = ~np.all(main_pixels == [0, 0, 0], axis=-1)
main_pixels_reshaped = main_pixels.reshape(-1, 3)

predicted = np.full((main_pixels_reshaped.shape[0],), -1)

# Previsão em blocos com barra de progresso
valid_indices = np.where(mask_valid.reshape(-1))[0]
batch_size = 5000
for i in tqdm(range(0, len(valid_indices), batch_size), desc="Classificando pixels"):
    batch_idx = valid_indices[i:i+batch_size]
    batch_pixels = main_pixels_reshaped[batch_idx]
    predicted[batch_idx] = clf.predict(batch_pixels)

predicted_img = predicted.reshape(height, width)

# === 6. Relatório ===
report = classification_report(y_train_sampled, clf.predict(X_train_sampled), target_names=["Estrada", "Vegetação", "Terra"])
print("Relatório de Classificação:\n", report)

# === 7. Reconstrução da imagem classificada ===
color_map = {
    -1: [0, 0, 0],        # Fundo preto
     0: [255, 255, 0],  # Estrada
     1: [139, 69, 19],    # Vegetação
     2: [70, 130, 180]     # Terra
}

segmented_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 3), dtype=np.uint8)
for class_val, color in color_map.items():
    segmented_img[predicted_img == class_val] = color

# === 8. Restaurar imagem para resolução original ===
segmented_img_original_size = Image.fromarray(segmented_img).resize(img_main_original.size)

# === 9. Mostrar e salvar ===
plt.figure(figsize=(10, 10))
plt.imshow(segmented_img_original_size)
plt.title("Imagem Classificada com KNN (resolução original)")
plt.axis("off")
plt.show()

segmented_img_original_size.save("imagem_classificada_knn_final.png")
