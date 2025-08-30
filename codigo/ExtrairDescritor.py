import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm

# Configurar numpy para exibir todos os valores do array
np.set_printoptions(threshold=np.inf)

# Carrega modelo pré-treinado (sem camada de classificação)
modelo = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def gerar_descritor_resnet(caminho_imagem):
    # Carregar imagem e redimensionar para 224x224 (entrada da ResNet)
    img = image.load_img(caminho_imagem, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Pré-processamento padrão da ResNet

    # Extrair características
    features = modelo.predict(x)
    return features.flatten()

# ============================
# 🔹 TESTE
# ============================

# Coloque aqui o caminho da imagem
caminho1 = "uff.jpeg"
caminho2 = "eu.jpeg"
caminho3 = "teste.jpeg"

d1 = gerar_descritor_resnet(caminho1)
d2 = gerar_descritor_resnet(caminho2)
d3 = gerar_descritor_resnet(caminho3)


# Distância Euclidiana
dist_euclidiana12 = norm(d1 - d2)
dist_euclidiana13 = norm(d1 - d3)
dist_euclidiana23 = norm(d2 - d3)

print("Distância entre as imagens 1 - 2:", dist_euclidiana12)
print("Distância entre as imagens 1 - 3:", dist_euclidiana13)
print("Distância entre as imagens 2 - 3:", dist_euclidiana23)