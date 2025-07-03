from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import pandas as pd

# === 1. Carregar imagens ===
img_main = Image.open("BSB-1.jpg").convert("RGB")
img_road = Image.open("teste_1.png").convert("RGB")        # Estrada
img_vegetation = Image.open("teste_2.png").convert("RGB")  # Vegetação
img_soil = Image.open("teste_3.png").convert("RGB")        # Terra

# === 2. Extrair pixels de cada amostra e rotular ===
def get_pixels_and_labels(image, label):
    pixels = np.array(image).reshape(-1, 3)
    labels = np.full((pixels.shape[0],), label)
    return pixels, labels

X_road, y_road = get_pixels_and_labels(img_road, 0)
X_veg, y_veg = get_pixels_and_labels(img_vegetation, 1)
X_soil, y_soil = get_pixels_and_labels(img_soil, 2)

# === 3. Unir dados de treino ===
X_train = np.vstack((X_road, X_veg, X_soil))
y_train = np.concatenate((y_road, y_veg, y_soil))

# === 4. Treinar modelo ===
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
clf.fit(X_train, y_train)

# === 5. Classificar imagem principal ===
main_pixels = np.array(img_main)                           # (H, W, 3)
height, width = img_main.size[1], img_main.size[0]
mask_valid = ~np.all(main_pixels == [0, 0, 0], axis=-1)     # Máscara onde NÃO é preto
main_pixels_reshaped = main_pixels.reshape(-1, 3)

# Inicializar como não classificado (-1)
predicted = np.full((main_pixels_reshaped.shape[0],), -1)

# Classificar apenas os pixels válidos
valid_pixels = main_pixels_reshaped[mask_valid.reshape(-1)]
predicted_valid = clf.predict(valid_pixels)
predicted[mask_valid.reshape(-1)] = predicted_valid

# Reshape para imagem 2D
predicted_img = predicted.reshape(height, width)

# === 6. Gerar relatório ===
report = classification_report(y_train, clf.predict(X_train), target_names=["Estrada", "Vegetação", "Terra"])
print("Relatório de Classificação:\n", report)

# === 7. Mostrar árvore de decisão ===
plt.figure(figsize=(16, 8))
plot_tree(clf, filled=True, feature_names=["R", "G", "B"], class_names=["Estrada", "Vegetação", "Terra"])
plt.title("Árvore de Decisão (baseada em Entropia)")
plt.tight_layout()
plt.show()

# === 8. Reconstruir imagem segmentada com cores ===
color_map = {
    -1: [0, 0, 0],        # Fundo preto (não classificado)
     0: [255, 255, 255],  # Estrada
     1: [34, 139, 34],    # Vegetação
     2: [160, 82, 45]     # Terra
}

segmented_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 3), dtype=np.uint8)
for class_val, color in color_map.items():
    segmented_img[predicted_img == class_val] = color

# Mostrar imagem final
plt.figure(figsize=(10, 10))
plt.imshow(segmented_img)
plt.title("Imagem Reconstruída com Classificação")
plt.axis("off")
plt.show()

# === 9. Salvar imagem classificada ===
Image.fromarray(segmented_img).save("imagem_classificada.png")
