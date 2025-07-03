import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar as imagens em escala de cinza
img1 = cv2.imread('BSB-1.jpg', cv2.IMREAD_GRAYSCALE)  # imagem de referência
img2 = cv2.imread('BSB.jpg', cv2.IMREAD_GRAYSCALE)    # imagem onde você quer encontrar a referência

# 2. Verificar se as imagens foram carregadas corretamente
if img1 is None or img2 is None:
    print("Erro ao carregar as imagens. Verifique os nomes e caminhos.")
    exit()

# 3. Reduzir a resolução se forem muito grandes
MAX_SIZE = 800

def resize_image(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

img1 = resize_image(img1)
img2 = resize_image(img2)

# 4. Criar o detector SIFT com limite de keypoints
sift = cv2.SIFT_create(nfeatures=500)

# 5. Detectar keypoints e descritores
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 6. Fazer correspondência usando BFMatcher com k=2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# 7. Aplicar teste de razão de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Número de boas correspondências: {len(good_matches)}")

# 8. Se houver correspondências suficientes, encontrar homografia
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimar a homografia
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Obter os cantos da imagem de referência
    h, w = img1.shape
    pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Desenhar contorno da imagem reconhecida
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2_detected = cv2.polylines(img2_color, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
else:
    print("Não há correspondências suficientes.")
    img2_detected = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# 9. Mostrar correspondências
img_matches = cv2.drawMatches(img1, kp1, img2_detected, kp2, good_matches, None, flags=2)

plt.figure(figsize=(15, 8))
plt.imshow(img_matches)
plt.title('Reconhecimento com SIFT (otimizado)')
plt.axis('off')
plt.show()
