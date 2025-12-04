#importando as bibliotacas necessárias:
import numpy as np
import matplotlib.pyplot as plt

#definindo a função de ativação sigmoide e sua derivada:
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def sigmoide_derivada(f):
    return f * (1 - f)

#definindo a arquitetura da rede:
x_entrada = np.array([[0,0], [0,1], [1,0], [1,1]])
y_final_esperado = np.array([[0],[1],[1],[0]])
taxa_aprendizado = 2 * 1e-1 #A definir ainda.
n_epocas = 3 * int(1e4) #A definir ainda.
n_neuronios_ocultos = 2 #tá meio hardcoded, mas basicamente é isso.
n_neuronios_saida = 1   #tá meio hardcoded, mas basicamente é isso.

#definindo os parametros neceessários (inicialmente todos com valores aleatórios e que serão reajustados):
d_entradas = x_entrada.shape[1]
w_pesos_ocultos = np.random.uniform(low = -1, high = 1, size = (d_entradas, n_neuronios_ocultos))
w_pesos_saida = np.random.uniform(low = -1, high = 1, size = (n_neuronios_ocultos, n_neuronios_saida))
b_ocultos = np.random.uniform(low = -1, high = 1, size = (n_neuronios_ocultos))
b_saida = np.random.uniform(low = -1, high = 1, size = (n_neuronios_saida))

#implementando treinamento da rede:
print("========== A rede está aprendendo: ==========")
for i in range (n_epocas):
    #========== Forward pass ==========:
    #Calcula pra cada neurônio o produto escalar e depois ativa esse valor com a função logística:
    somatorio_ocultos = np.dot(x_entrada, w_pesos_ocultos) + b_ocultos
    y_ocultos = sigmoide(somatorio_ocultos)
    #Calcula a saida no neurônio:
    somatorio_final = np.dot(y_ocultos, w_pesos_saida) + b_saida
    y_final = sigmoide(somatorio_final)
    
    #========== Backward pass ==========:
    erro_saida = y_final_esperado - y_final
    dy_final = sigmoide_derivada(y_final)
    delta_saida = erro_saida * dy_final
    #Propagação do delta (dy) nos outros neurônios):
    erro_oculto = np.dot(delta_saida, w_pesos_saida.T)
    dy_oculto = sigmoide_derivada(y_ocultos)
    delta_oculto = erro_oculto * dy_oculto
    
    #atualizando os pesos: 
        #Neurônio de saida
    dw_saida = np.dot(y_ocultos.T, delta_saida)
    w_pesos_saida = w_pesos_saida + (taxa_aprendizado * dw_saida)
    db_saida = np.sum(delta_saida)
    b_saida = b_saida + (taxa_aprendizado * db_saida)
        #Neurônios ocultos:
    dw_oculto = np.dot(x_entrada.T, delta_oculto)
    w_pesos_ocultos = w_pesos_ocultos + (taxa_aprendizado*dw_oculto)
    db_oculto = np.sum(delta_oculto, axis = 0)
    b_ocultos = b_ocultos + (taxa_aprendizado * db_oculto)
    
    #calcula e Imprime o custo (MSE):
    if i % 1000 == 0:  # Imprime a cada 100 épocas
        custo = np.mean(np.square(erro_saida))
        print(f"Época {i}, Custo (MSE): {custo:.6f}")
        
print("\n========== O que a rede aprendeu? ==========")
#executa o Forward Pass com os pesos finais:
somatorio_ocultos = np.dot(x_entrada, w_pesos_ocultos) + b_ocultos
y_ocultos = sigmoide(somatorio_ocultos)
somatorio_final = np.dot(y_ocultos, w_pesos_saida) + b_saida
y_final = sigmoide(somatorio_final)

print("Gabarito (Y Esperado):")
print(y_final_esperado)
print("\nPrevisão Final (Y Calculado):")
print(y_final)

#geração de gráficos

print("\n========== Gerando Gráficos... ==========")

# Configurações visuais
plt.figure(figsize=(12, 5)) # Cria uma figura larga para colocar os 2 gráficos lado a lado

#GRÁFICO 1: O Problema XOR em si
plt.subplot(1, 2, 1) #(1 linha, 2 colunas, gráfico nº 1)
plt.title("1. O Problema XOR (Alvo)")
plt.xlabel("Entrada X1")
plt.ylabel("Entrada X2")

# Plota os pontos: Vermelho para 0, Azul para 1
# c=y_final_esperado.ravel() usa os valores 0 e 1 para escolher a cor
plt.scatter(x_entrada[:,0], x_entrada[:,1], c=y_final_esperado.ravel(), 
            cmap='bwr', s=200, edgecolors='k')
plt.grid(True, linestyle='--', alpha=0.6)


# GRÁFICO 2: O Resultado (Fronteira de Decisão) 
plt.subplot(1, 2, 2) # (1 linha, 2 colunas, gráfico nº 2)
plt.title("2. O Resultado (Fronteira de Decisão)")
plt.xlabel("Entrada X1")
plt.ylabel("Entrada X2")

# Criar uma "grade" (meshgrid) de pontos cobrindo todo o fundo do gráfico
h = 0.01 # Resolução da grade
x_min, x_max = x_entrada[:, 0].min() - 0.5, x_entrada[:, 0].max() + 0.5
y_min, y_max = x_entrada[:, 1].min() - 0.5, x_entrada[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Achatar a grade para passar pela rede (como se fossem milhares de entradas)
pontos_grade = np.c_[xx.ravel(), yy.ravel()]

# FORWARD PASS na grade (usando os pesos treinados!)
# Camada Oculta
grade_oculta = np.dot(pontos_grade, w_pesos_ocultos) + b_ocultos
y_grade_oculta = sigmoide(grade_oculta)
# Camada de Saída
grade_final = np.dot(y_grade_oculta, w_pesos_saida) + b_saida
y_grade_final = sigmoide(grade_final)

# Redimensionar a previsão de volta para o formato da grade (para pintar o fundo)
Z = y_grade_final.reshape(xx.shape)

# Níveis de contorno para ver a transição suave
plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.8) 
plt.colorbar(label='Probabilidade de ser 1')

# Plotar os pontos originais por cima para comparar
plt.scatter(x_entrada[:,0], x_entrada[:,1], c=y_final_esperado.ravel(), 
            cmap='bwr', s=200, edgecolors='k')

plt.tight_layout() # Ajusta o espaçamento
plt.show()