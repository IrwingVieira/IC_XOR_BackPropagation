# Implementação de MLP com Backpropagation - Problema XOR

Este repositório contém o código desenvolvido durante o projeto de Iniciação Científica sobre Redes Neurais Artificiais. O objetivo principal foi implementar um **Multilayer Perceptron (MLP)** para resolver o problema não-linearmente separável do **XOR**.

## 🎯 Objetivo
Demonstrar e compreender matematicamente o funcionamento do algoritmo **Backpropagation** e como camadas ocultas permitem que uma rede neural resolva problemas que modelos lineares (como Perceptron simples e Adaline) não conseguem.

## 🛠️ Tecnologias Utilizadas
* **Python 3**
* **NumPy:** Para todas as operações matriciais e álgebra linear (dot product, transposição, etc).
* **Matplotlib:** Para visualização dos dados e plotagem da fronteira de decisão.

## 🧠 Arquitetura da Rede
A rede implementada possui a seguinte topologia:
* **Camada de Entrada:** 2 neurônios (Entradas $x_1$ e $x_2$).
* **Camada Oculta:** 2 neurônios (Ativação Sigmoide).
* **Camada de Saída:** 1 neurônio (Ativação Sigmoide).

## 🚀 Como Executar

1. Certifique-se de ter o Python e as bibliotecas instaladas:
   ```bash
   pip install numpy matplotlib