from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import pandas as pd

# === 1. Carregar imagens ===
img_main = Image.open("4.jpg").convert("RGB")
img_road = Image.open("teste_1.jpeg").convert("RGB")        # Estrada
img_vegetation = Image.open("teste_2.jpeg").convert("RGB")  # Vegetação
img_soil = Image.open("teste_3.jpeg").convert("RGB")        # Terra

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
main_pixels = np.array(img_main).reshape(-1, 3)
predicted = clf.predict(main_pixels)
predicted_img = predicted.reshape(img_main.size[1], img_main.size[0])  # (altura, largura)

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
    0: [255, 255, 255],  # Estrada - Branco
    1: [34, 139, 34],    # Vegetação - Verde escuro
    2: [160, 82, 45]     # Terra - Marrom
}

segmented_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 3), dtype=np.uint8)
for class_val, color in color_map.items():
    segmented_img[predicted_img == class_val] = color

plt.figure(figsize=(10, 10))
plt.imshow(segmented_img)
plt.title("Imagem Reconstruída com Classificação")
plt.axis("off")
plt.show()

# === 9. (Opcional) Salvar imagem classificada ===
Image.fromarray(segmented_img).save("imagem_classificada.jpeg")

