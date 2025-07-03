import numpy as np
import pandas as pd
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import time
from numba import jit, cuda
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
warnings.filterwarnings('ignore')

# Função otimizada para extração de características com Numba
@jit(nopython=True)
def extract_features_numba(rgb_array):
    """
    Extração de características otimizada com Numba
    """
    n_samples = rgb_array.shape[0]
    features = np.zeros((n_samples, 19))
    
    for i in range(n_samples):
        r, g, b = rgb_array[i, 0], rgb_array[i, 1], rgb_array[i, 2]
        
        # Características básicas
        features[i, 0] = r
        features[i, 1] = g  
        features[i, 2] = b
        
        # Intensidade
        features[i, 3] = (r + g + b) / 3
        
        # Saturação
        features[i, 4] = max(r, g, b) - min(r, g, b)
        
        # Razões (com proteção contra divisão por zero)
        features[i, 5] = r / (g + 1e-8)
        features[i, 6] = r / (b + 1e-8)
        features[i, 7] = g / (b + 1e-8)
        
        # Índices
        features[i, 8] = (g - r) / (g + r + 1e-8)  # Vegetation index
        features[i, 9] = (r - g) / (r + g + b + 1e-8)  # Soil index
        
        # Diferenças
        features[i, 10] = abs(r - g)
        features[i, 11] = abs(r - b)
        features[i, 12] = abs(g - b)
        
        # HSV simplificado
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Hue
        if diff == 0:
            hue = 0
        elif max_val == r:
            hue = 60 * ((g - b) / diff)
        elif max_val == g:
            hue = 60 * (2 + (b - r) / diff)
        else:
            hue = 60 * (4 + (r - g) / diff)
        
        features[i, 13] = hue
        features[i, 14] = 0 if max_val == 0 else diff / max_val * 100  # Saturation
        features[i, 15] = max_val  # Value
        
        # Brilho e contraste
        features[i, 16] = max_val
        features[i, 17] = min_val
        features[i, 18] = diff
    
    return features

class RGBImageClassifier:
    def __init__(self, dt_params=None, knn_params=None, mlp_params=None):
        """
        Classificador de imagens RGB otimizado para alta performance
        """
        # Parâmetros otimizados
        if dt_params is None:
            dt_params = {'max_depth': 10, 'random_state': 42, 'min_samples_split': 10}
        
        if knn_params is None:
            knn_params = {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean', 
                         'algorithm': 'kd_tree'}  # Algoritmo mais rápido
        
        if mlp_params is None:
            mlp_params = {
                'hidden_layer_sizes': (64, 32), 
                'max_iter': 1000, 
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        
        self.dt_classifier = DecisionTreeClassifier(**dt_params)
        self.knn_classifier = KNeighborsClassifier(**knn_params)
        self.mlp_classifier = MLPClassifier(**mlp_params)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.class_names = ['Vegetação', 'Pasto', 'Pista', 'Construção']
        self.class_colors = {
            'Vegetação': [0, 255, 0],
            'Pasto': [144, 238, 144],
            'Pista': [128, 128, 128],
            'Construção': [255, 0, 0]
        }
        
        self.is_fitted = False
        self.feature_names = [
            'R', 'G', 'B', 'Intensity', 'Saturation', 'RG_Ratio', 'RB_Ratio', 
            'GB_Ratio', 'Vegetation_Index', 'Soil_Index', 'RG_Diff', 'RB_Diff', 'GB_Diff',
            'Hue', 'Sat_HSV', 'Value_HSV', 'Brightness', 'Darkness', 'Contrast'
        ]
        
    def load_csv_data(self, csv_paths):
        """Carrega dados dos CSVs"""
        print("=== CARREGANDO DADOS DOS CSVs ===")
        
        all_rgb_data = []
        all_labels = []
        
        for class_name, csv_path in csv_paths.items():
            try:
                print(f"Carregando {class_name}...")
                df = pd.read_csv(csv_path)
                
                # Verificar diferentes formatos de colunas possíveis
                if 'R' in df.columns and 'G' in df.columns and 'B' in df.columns:
                    rgb_data = df[['R', 'G', 'B']].values
                elif 'r' in df.columns and 'g' in df.columns and 'b' in df.columns:
                    rgb_data = df[['r', 'g', 'b']].values
                elif 'Red' in df.columns and 'Green' in df.columns and 'Blue' in df.columns:
                    rgb_data = df[['Red', 'Green', 'Blue']].values
                else:
                    # Usar as 3 primeiras colunas numéricas
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 3:
                        rgb_data = df[numeric_cols[:3]].values
                    else:
                        print(f"❌ Erro: CSV {csv_path} não possui colunas RGB válidas")
                        continue
                
                # Limpar dados inválidos
                valid_mask = ~np.isnan(rgb_data).any(axis=1)
                rgb_data = rgb_data[valid_mask]
                
                # Garantir que os valores estão no range [0, 255]
                rgb_data = np.clip(rgb_data, 0, 255)
                
                all_rgb_data.extend(rgb_data.tolist())
                all_labels.extend([class_name] * len(rgb_data))
                
                print(f"✓ {class_name}: {len(rgb_data)} amostras válidas")
                
            except FileNotFoundError:
                print(f"❌ Arquivo não encontrado: {csv_path}")
                continue
            except Exception as e:
                print(f"❌ Erro ao carregar {csv_path}: {e}")
                continue
        
        if len(all_rgb_data) == 0:
            raise ValueError("Nenhum dado válido foi carregado dos CSVs!")
        
        print(f"✅ Total: {len(all_rgb_data)} amostras carregadas")
        return np.array(all_rgb_data), np.array(all_labels)
    
    def fit(self, csv_paths):
        """Treina o modelo"""
        print("=== TREINAMENTO OTIMIZADO ===")
        
        # Carregar dados
        X_rgb, y_classes = self.load_csv_data(csv_paths)
        
        # Extração otimizada
        print("Extraindo características (otimizado com Numba)...")
        start_time = time.time()
        X_features = extract_features_numba(X_rgb.astype(np.float64))
        extraction_time = time.time() - start_time
        print(f"✓ Características extraídas em {extraction_time:.2f}s")
        
        # Dividir dados
        y_encoded = self.label_encoder.fit_transform(y_classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar modelos
        print("Treinando modelos...")
        train_start = time.time()
        
        self.dt_classifier.fit(X_train_scaled, y_train)
        print("✓ Decision Tree treinado")
        
        self.knn_classifier.fit(X_train_scaled, y_train)
        print("✓ KNN treinado")
        
        # Preparar dados para MLP (ensemble)
        dt_pred_train = self.dt_classifier.predict(X_train_scaled)
        knn_pred_train = self.knn_classifier.predict(X_train_scaled)
        ensemble_pred_train = self.voting_mechanism(dt_pred_train, knn_pred_train)
        
        mlp_train_input = np.column_stack([X_train_scaled, ensemble_pred_train.reshape(-1, 1)])
        self.mlp_classifier.fit(mlp_train_input, y_train)
        print("✓ MLP treinado")
        
        train_time = time.time() - train_start
        
        # Avaliar modelo
        dt_pred_test = self.dt_classifier.predict(X_test_scaled)
        knn_pred_test = self.knn_classifier.predict(X_test_scaled)
        ensemble_pred_test = self.voting_mechanism(dt_pred_test, knn_pred_test)
        mlp_test_input = np.column_stack([X_test_scaled, ensemble_pred_test.reshape(-1, 1)])
        mlp_pred_test = self.mlp_classifier.predict(mlp_test_input)
        
        mlp_acc = accuracy_score(y_test, mlp_pred_test)
        
        print(f"\n✅ TREINAMENTO CONCLUÍDO")
        print(f"   Tempo de extração: {extraction_time:.2f}s")
        print(f"   Tempo de treinamento: {train_time:.2f}s")
        print(f"   Acurácia Final: {mlp_acc:.3f} ({mlp_acc*100:.1f}%)")
        
        # Relatório detalhado
        class_names_decoded = self.label_encoder.inverse_transform(range(len(self.label_encoder.classes_)))
        print(f"\n📊 Relatório de Classificação:")
        print(classification_report(y_test, mlp_pred_test, 
                                  target_names=class_names_decoded, 
                                  digits=3))
        
        self.is_fitted = True
        return {
            'accuracy': mlp_acc,
            'extraction_time': extraction_time,
            'training_time': train_time,
            'total_samples': len(X_rgb)
        }
    
    def voting_mechanism(self, dt_predictions, knn_predictions):
        """Voting otimizado vetorizado"""
        return np.where(dt_predictions == knn_predictions, dt_predictions, knn_predictions)
    
    def predict_image(self, image_path, batch_size=50000, downsample=None):
        """
        Classificação otimizada com processamento em lotes
        Método compatível com a interface original
        """
        return self.predict_image_optimized(image_path, batch_size, downsample)
    
    def predict_image_optimized(self, image_path, batch_size=50000, downsample=None):
        """
        Classificação otimizada com processamento em lotes e redimensionamento
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado! Execute fit() primeiro.")
        
        print(f"=== CLASSIFICAÇÃO OTIMIZADA ===")
        print(f"Arquivo: {image_path}")
        
        # Carregar imagem
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        except Exception as e:
            raise ValueError(f"Erro ao carregar imagem {image_path}: {e}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape
        
        # Redimensionar se necessário
        if downsample and (original_shape[0] > downsample or original_shape[1] > downsample):
            # Manter proporção
            scale = downsample / max(original_shape[:2])
            new_width = int(original_shape[1] * scale)
            new_height = int(original_shape[0] * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            print(f"Imagem redimensionada: {original_shape[:2]} → {image_rgb.shape[:2]}")
        
        print(f"Processando: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixels")
        
        # Processar pixels
        pixels = image_rgb.reshape(-1, 3).astype(np.float64)
        total_pixels = len(pixels)
        
        print(f"Total de pixels: {total_pixels:,}")
        
        # Processamento em lotes
        all_predictions = []
        start_time = time.time()
        
        for i in range(0, total_pixels, batch_size):
            batch_end = min(i + batch_size, total_pixels)
            batch_pixels = pixels[i:batch_end]
            
            # Extração otimizada com Numba
            batch_features = extract_features_numba(batch_pixels)
            batch_features_scaled = self.scaler.transform(batch_features)
            
            # Predições ensemble
            dt_pred = self.dt_classifier.predict(batch_features_scaled)
            knn_pred = self.knn_classifier.predict(batch_features_scaled)
            ensemble_pred = self.voting_mechanism(dt_pred, knn_pred)
            
            mlp_input = np.column_stack([batch_features_scaled, ensemble_pred.reshape(-1, 1)])
            final_pred = self.mlp_classifier.predict(mlp_input)
            
            all_predictions.extend(final_pred)
            
            # Progresso
            progress = (batch_end / total_pixels) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / progress * 100) - elapsed if progress > 0 else 0
            print(f"Progresso: {progress:.1f}% | Tempo: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')
        
        processing_time = time.time() - start_time
        print(f"\n✅ Classificação concluída em {processing_time:.1f}s")
        
        # Decodificar e reconstruir
        final_classes = self.label_encoder.inverse_transform(all_predictions)
        classified_image = self.reconstruct_classified_image(final_classes, image_rgb.shape)
        
        # Redimensionar resultado para tamanho original se necessário
        if downsample and image_rgb.shape != original_shape:
            classified_image = cv2.resize(classified_image, (original_shape[1], original_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            print("Resultado redimensionado para tamanho original")
        
        # Estatísticas
        class_counts = dict(Counter(final_classes))
        total_pixels_final = len(final_classes)
        
        print("\n📊 Distribuição das Classes:")
        for class_name, count in class_counts.items():
            percentage = (count / total_pixels_final) * 100
            print(f"   {class_name}: {count:,} pixels ({percentage:.1f}%)")
        
        return {
            'original_image': cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
            'classified_image': classified_image,
            'class_counts': class_counts,
            'processing_time': processing_time,
            'image_shape': original_shape
        }
    
    def reconstruct_classified_image(self, predictions, shape):
        """Reconstrói imagem classificada otimizada"""
        h, w = shape[:2]
        classified_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        predictions_2d = predictions.reshape(h, w)
        
        for class_name in self.label_encoder.classes_:
            mask = predictions_2d == class_name
            color = self.class_colors.get(class_name, [255, 255, 255])
            classified_img[mask] = color
        
        return classified_img
    
    def visualize_results(self, result):
        """Visualização dos resultados"""
        plt.figure(figsize=(15, 5))
        
        # Imagem original
        plt.subplot(1, 3, 1)
        plt.imshow(result['original_image'])
        plt.title('Imagem Original')
        plt.axis('off')
        
        # Imagem classificada
        plt.subplot(1, 3, 2)
        plt.imshow(result['classified_image'])
        plt.title(f'Classificada ({result["processing_time"]:.1f}s)')
        plt.axis('off')
        
        # Gráfico de distribuição
        plt.subplot(1, 3, 3)
        classes = list(result['class_counts'].keys())
        counts = list(result['class_counts'].values())
        percentages = [count/sum(counts)*100 for count in counts]
        colors = [np.array(self.class_colors.get(c, [128, 128, 128]))/255 for c in classes]
        
        bars = plt.bar(classes, percentages, color=colors)
        plt.title('Distribuição das Classes')
        plt.ylabel('Porcentagem (%)')
        plt.xticks(rotation=45)
        
        # Adicionar valores nas barras
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar legenda de cores
        print("\n🎨 Legenda de Cores:")
        for class_name, color in self.class_colors.items():
            print(f"   {class_name}: RGB{color}")


# ========== EXECUÇÃO PRINCIPAL ==========

if __name__ == "__main__":
    print("🚀 CLASSIFICADOR RGB OTIMIZADO")
    print("="*50)
    
    # 1. Configurar caminhos dos CSVs
    csv_paths = {
        'Vegetação': 'csv/pixels_rgb_rgb.csv',
        'Pasto': 'csv/pasto.csv',
        'Pista': 'csv/pista.csv',
        'Construção': 'csv/construcao.csv'
    }
    
    try:
        # 2. Criar e treinar o classificador
        classifier = RGBImageClassifier()
        results = classifier.fit(csv_paths)
        
        # 3. Classificar imagem
        print("\n" + "="*50)
        result = classifier.predict_image('BSB-1.jpg')
        
        # 4. Visualizar resultados
        classifier.visualize_results(result)
        
        # 5. Resumo final
        print("\n" + "="*50)
        print("🎯 RESUMO DA EXECUÇÃO:")
        print(f"   ✅ Modelo treinado com {results['total_samples']:,} amostras")
        print(f"   ✅ Acurácia: {results['accuracy']:.1%}")
        print(f"   ✅ Imagem classificada em {result['processing_time']:.1f}s")
        print(f"   ✅ Resolução: {result['image_shape'][1]}x{result['image_shape'][0]} pixels")
        print("\n🔧 OTIMIZAÇÕES ATIVAS:")
        print("   • Extração de características com Numba (JIT)")
        print("   • Processamento em lotes")
        print("   • Algoritmo KNN otimizado (kd_tree)")
        print("   • Voting mechanism vetorizado")
        print("   • Early stopping no MLP")
        print("="*50)
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        print("\n🔍 VERIFICAÇÕES:")
        print("   1. Os arquivos CSV existem nos caminhos especificados?")
        print("   2. A imagem 'BSB-1.jpg' existe no diretório atual?")
        print("   3. As dependências estão instaladas? (pip install numba)")
        print("   4. Os CSVs possuem colunas R, G, B válidas?")