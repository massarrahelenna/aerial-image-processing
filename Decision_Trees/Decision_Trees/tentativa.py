from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier 
from sklearn.metrics import classification_report
from sklearn.utils import resample
from tqdm import tqdm

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

# === 3. Reduzir para 10.000 amostras balanceadas ===
X_train_sampled, y_train_sampled = resample(X_train, y_train, n_samples=10000, stratify=y_train, random_state=42)

# === 4. Treinar classificadores ===
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=10)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)

knn.fit(X_train_sampled, y_train_sampled)
dt.fit(X_train_sampled, y_train_sampled)
mlp.fit(X_train_sampled, y_train_sampled)

# === 5. Ensemble: Voting Classifier ===
ensemble = VotingClassifier(estimators=[
    ('knn', knn),
    ('dt', dt),
    ('mlp', mlp)
], voting='hard')
ensemble.fit(X_train_sampled, y_train_sampled)

# === 6. Avaliar modelos ===
print("\n--- Avaliação Individual ---")
print("KNN:\n", classification_report(y_train_sampled, knn.predict(X_train_sampled)))
print("Decision Tree:\n", classification_report(y_train_sampled, dt.predict(X_train_sampled)))
print("MLP:\n", classification_report(y_train_sampled, mlp.predict(X_train_sampled)))
print("\n--- Ensemble ---")
print("Voting Classifier:\n", classification_report(y_train_sampled, ensemble.predict(X_train_sampled)))

# === 7. Função para classificar imagem com qualquer modelo ===
def classify_image(model, img_main, color_map):
    main_pixels = np.array(img_main)
    height, width = img_main.size[1], img_main.size[0]
    mask_valid = ~np.all(main_pixels == [0, 0, 0], axis=-1)
    main_pixels_reshaped = main_pixels.reshape(-1, 3)
    
    predicted = np.full((main_pixels_reshaped.shape[0],), -1)
    valid_indices = np.where(mask_valid.reshape(-1))[0]
    
    batch_size = 5000
    for i in tqdm(range(0, len(valid_indices), batch_size), desc="Classificando pixels"):
        batch_idx = valid_indices[i:i+batch_size]
        batch_pixels = main_pixels_reshaped[batch_idx]
        predicted[batch_idx] = model.predict(batch_pixels)

    predicted_img = predicted.reshape(height, width)
    segmented_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 3), dtype=np.uint8)
    for class_val, color in color_map.items():
        segmented_img[predicted_img == class_val] = color
    return segmented_img

# === 8. Mapa de cores ===
color_map = {
    -1: [0, 0, 0],        # Fundo
     0: [255, 255, 0],    # Estrada
     1: [34, 139, 34],    # Vegetação
     2: [160, 82, 45]     # Terra
}

# === 9. Classificar e salvar imagens ===
models = {'knn': knn, 'dt': dt, 'mlp': mlp, 'ensemble': ensemble}
for name, model in models.items():
    segmented = classify_image(model, img_main, color_map)
    segmented_resized = Image.fromarray(segmented).resize(img_main_original.size)
    segmented_resized.save(f"classificada_{name}.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_resized)
    plt.title(f"Segmentação: {name.upper()}")
    plt.axis("off")
    plt.show()
 
