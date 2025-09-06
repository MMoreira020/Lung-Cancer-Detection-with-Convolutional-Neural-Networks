# Lung-Cancer-Detection-with-Convolutional-Neural-Networks
Análise comparativa de 5 arquiteturas de Redes Neurais (ResNet, DenseNet, Inception, SqueezeNet, EfficientNet) treinadas com otimizadores ADAM e SGD. Este projeto visa demonstrar o impacto da escolha do otimizador na acurácia e na velocidade de treinamento dos modelos.

## 🧠 Modelos

As seguintes arquiteturas de redes neurais foram implementadas e treinadas:

* **ResNet-50:** Uma rede neural convolucional de 50 camadas que introduz o conceito de "conexões residuais" (skip connections) para mitigar o problema do gradiente descendente e permitir o treinamento de redes muito profundas.

* **DenseNet-121:** Caracteriza-se por sua conectividade densa, onde cada camada recebe as saídas de todas as camadas anteriores. Isso promove a reutilização de características, melhora o fluxo de gradiente e reduz o número de parâmetros.

* **InceptionV3:** Desenvolvida pelo Google, esta arquitetura utiliza "módulos Inception" que aplicam convoluções de diferentes tamanhos em paralelo, permitindo a captura de características em múltiplas escalas de forma computacionalmente eficiente.

* **SqueezeNet 1.0:** Uma arquitetura leve e eficiente, projetada para ter um número significativamente menor de parâmetros em comparação com outras redes, sem comprometer a acurácia. É ideal para aplicações em dispositivos com recursos limitados.

* **EfficientNetB2:** Parte de uma família de modelos que foram escalonados de forma otimizada em termos de profundidade, largura e resolução da rede através de um coeficiente composto. O EfficientNetB2 oferece um excelente equilíbrio entre acurácia e eficiência computacional.

## ⚙️ Otimizadores

Cada um dos modelos acima foi treinado utilizando os dois otimizadores a seguir:

* **ADAM (Adaptive Moment Estimation):** Um otimizador popular que calcula taxas de aprendizado adaptativas para cada parâmetro. Ele combina as vantagens de dois outros algoritmos, Momentum e RMSprop, e é conhecido por sua rápida convergência.

* **SGD (Stochastic Gradient Descent):** Um otimizador clássico e fundamental. A versão utilizada neste projeto provavelmente inclui aprimoramentos comuns como momentum e decaimento da taxa de aprendizado, que ajudam a acelerar o treinamento e a evitar mínimos locais.

## 📂 Estrutura do Repositório

├── DenseNet-121/
│   ├── Bruto/
│   │   ├── DenseNet-121_ADAM.py
│   │   └── DenseNet-121_SGD.py
│   └── Filtro_de_Wavelet/
│       ├── DenseNet-121_ADAM.py
│       └── DenseNet-121_SGD.py
├── EfficientNetB2/
│   ├── Bruto/
│   │   ├── EfficientNetB2_ADAM.py
│   │   └── EfficientNetB2_SGD.py
│   └── Filtro_de_Wavelet/
│       ├── EfficientNetB2_ADAM.py
│       └── EfficientNetB2_SGD.py
├── InceptionV3/
│   ├── Bruto/
│   │   ├── InceptionV3_ADAM.py
│   │   └── InceptionV3_SGD.py
│   └── Filtro_de_Wavelet/
│       ├── InceptionV3_ADAM.py
│       └── InceptionV3_SGD.py
├── ResNet-50/
│   ├── Bruto/
│   │   ├── ResNet-50_ADAM.py
│   │   └── ResNet-50_SGD.py
│   └── Filtro_de_Wavelet/
│       ├── ResNet-50_ADAM.py
│       └── ResNet-50_SGD.py
├── SqueezeNet1_0/
│   ├── Bruto/
│   │   ├── SqueezeNet1_0_ADAM.py
│   │   └── SqueezeNet1_0_SGD.py
│   └── Filtro_de_Wavelet/
│       ├── SqueezeNet1_0_ADAM.py
│       └── SqueezeNet1_0_SGD.py
└── README.md

## 🚀 Como Utilizar

Para executar os scripts de treinamento, você precisará ter as seguintes bibliotecas instaladas:

* Python 3.x
* PyTorch
* NumPy
* Matplotlib (para visualização dos resultados)

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/Lung-Cancer-Detection-with-Convolutional-Neural-Networks.git
    ```

2.  **Navegue até a pasta do modelo desejado:**
    ```bash
    # Exemplo para treinar ResNet-50 com dados brutos e otimizador ADAM
    cd Lung-Cancer-Detection-with-Convolutional-Neural-Networks/ResNet-50/Bruto

    # Exemplo para treinar ResNet-50 com dados filtrados (Wavelet) e otimizador SGD
    cd Lung-Cancer-Detection-with-Convolutional-Neural-Networks/ResNet-50/Filtro_de_Wavelet
    ```

3.  **Execute o script de treinamento:**
    ```bash
    # Exemplo para o primeiro caso (ResNet-50, Bruto, ADAM)
    python ResNet-50_ADAM.py

    # Exemplo para o segundo caso (ResNet-50, Filtro_de_Wavelet, SGD)
    python ResNet-50_SGD.py
    ```

## 📈 Resultados (Exemplo)

*(Esta seção apresenta um resumo comparativo da acurácia de validação para cada arquitetura, utilizando os dados do conjunto de teste original.)*

| Modelo          | Otimizador | Acurácia de Validação |
| :-------------- | :--------- | :-------------------- |
| ResNet-50       | ADAM       | 92.78%                |
| ResNet-50       | SGD        | 91.72%                |
| DenseNet-121    | ADAM       | 90.00%                |
| DenseNet-121    | SGD        | 91.10%                |
| EfficientNet-B2 | ADAM       | 88.67%                |
| EfficientNet-B2 | SGD        | 90.41%                |
| Inception.V3    | ADAM       | 86.62%                |
| Inception.V3    | SGD        | 81.82%                |
| Squeezenet1.0   | ADAM       | 90.41%                |
| Squeezenet1.0   | SGD        | 86.56%                |

*(Esta seção apresenta um resumo comparativo da acurácia de validação para cada arquitetura, utilizando os dados do conjunto de teste processado com Wavelet.)*

| Modelo          | Otimizador | Acurácia de Validação |
| :-------------- | :--------- | :-------------------- |
| ResNet-50       | ADAM       | 90.41%                |
| ResNet-50       | SGD        | 91.10%                |
| DenseNet-121    | ADAM       | 90.41%                |
| DenseNet-121    | SGD        | 90.73%                |
| EfficientNet-B2 | ADAM       | 86.30%                |
| EfficientNet-B2 | SGD        | 91.10%                |
| Inception.V3    | ADAM       | 83.30%                |
| Inception.V3    | SGD        | 85.93%                |
| Squeezenet1.0   | ADAM       | 90.56%                |
| Squeezenet1.0   | SGD        | 91.10%                |


## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões para melhorar este projeto, sinta-se à vontade para abrir uma "issue" ou enviar um "pull request".
