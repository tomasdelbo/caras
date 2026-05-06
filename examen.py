"""
examen2.py — BackPropagation Región del Plano.

Algoritmo de BackPropagation manual (red feedforward + retropropagación online,
sin frameworks) sobre los embeddings ISOMAP del trabajo de caras.

Incluye:
  * Algoritmo de BackPropagation (Región del Plano)
  * Código en Python del Algoritmo de BackPropagation
  * Código en Python que grafica cómo van evolucionando las rectas que separan
    a los positivos de los negativos (primera hidden layer).
"""

import csv
import math

import numpy as np
from sklearn.preprocessing import StandardScaler

from config import TEST_CUSTOM_TXT, TEST_TXT, TRAINING_TXT


# definicion de las funciones de activacion
#  y sus derivadas
#  ahora agregando las versiones VECTORIZADAS

def func_eval(fname, x):
    match fname:
        case "purelin":
            y = x
        case "logsig":
            y = 1.0 / ( 1.0 + math.exp(-x) )
        case "tansig":
            y = 2.0 / ( 1.0 + math.exp(-2.0*x) ) - 1.0
    return y

# version vectorizada de func_eval
func_eval_vec = np.vectorize(func_eval)


def deriv_eval(fname, y):  #atencion que y es la entrada y=f( x )
    match fname:
        case "purelin":
            d = 1.0
        case "logsig":
            d = y*(1.0-y)
        case "tansig":
            d = 1.0 - y*y
    return d


# version vectorizada de deriv_eval
deriv_eval_vec = np.vectorize(deriv_eval)


# Los datos que debe modelar la  Deep Neural Network
# Leo de mi archivo local de embeddings ISOMAP

with open(TRAINING_TXT, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    filas = list(reader)

y_idx = header.index("y")
x_idxs = [i for i in range(len(header)) if i != y_idx]

X = np.array([[float(r[i]) for i in x_idxs] for r in filas], dtype=np.float64)
clases_str = np.array([r[y_idx] for r in filas])

# One-hot encoding de las clases (multiclase)
clases_unicas = sorted(set(clases_str))
clase_a_idx = {c: i for i, c in enumerate(clases_unicas)}
Y = np.zeros((len(clases_str), len(clases_unicas)), dtype=np.float64)
for r, c in enumerate(clases_str):
    Y[r, clase_a_idx[c]] = 1.0

# Estandarizo X porque los embeddings ISOMAP tienen rangos grandes
# y con logsig se saturan las neuronas
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Arquitectura de la red
# Tamano datos

X_row = X.shape[0]
X_col = X.shape[1]

Y_col = Y.shape[1]


# Defino manualmente la arquitectura de la red
#  input es X_col e  Y_col

# DOS hidden layers
# el primer hidden layer tiene 6 perceptrones

arquitectura = {
   "input_size" : X_col,
   "layers_qty" : 2, # incluye la capa de salida, pero no la de entrada
   "layers_size" : [ 32, Y_col],
   "layers_func" : ['logsig','logsig','logsig'],
}


# seteo de la semilla aleatoria
np.random.seed(999979) # mi querida random seed para que las corridas sean reproducibles


# Inicializo la red con pesos al azar
#   a partir de la arquitectura

red = {
    'arq' : arquitectura,
    'layer' : list(),
}

niveles = red["arq"]["layers_qty"]

for i in range(niveles):
  nivel = dict()
  nivel["id"] = i
  nivel["last"] = (i==(niveles-1))
  nivel["size"] = red["arq"]["layers_size"][i]
  nivel["func"] = red["arq"]["layers_func"][i]

  if( i==0 ):
    entrada_size = red["arq"]["input_size"]
  else:
    entrada_size =  red["arq"]["layers_size"][i-1]

  salida_size =  nivel["size"]
  nivel["W"] = np.random.uniform(-0.5, 0.5, [salida_size, entrada_size])
  nivel["w0"] = np.random.uniform(-0.5, 0.5, [salida_size, 1])

  red["layer"].append(nivel)


# controles del entrenamiento

# Limite de lo que estoy dispuesto a trabajar
epoch_limit = 3000    # para terminar si no converge

# cuando la mejora del error sea inferior a este valor, me detendré
error_delta_umbral = 1.0e-07

# controla la velocidad de convergencia
learning_rate = 0.04


# inicializaciones del bucle principal del backpropagation

epoch = 0
error_epoch = float('inf')
error_ant =  0.0

# cada cuantos epochs imprimo el avance del entrenamiento
log_cada = 25

print("========= ENTRENAMIENTO =========")
print(f"Muestras de training: {X_row}  |  features: {X_col}  |  clases: {Y_col}")
print(f"Arquitectura: {arquitectura['layers_size']}  funcs: {arquitectura['layers_func']}")
print(f"epoch_limit={epoch_limit}  learning_rate={learning_rate}  error_delta_umbral={error_delta_umbral}")
print(f"{'epoch':>6} | {'error':>12} | {'delta':>12}")
print("-" * 40)


# el bucle principal del algoritmo BackPropagation

# continuo mientras en la iteración anterior modifique algo  y NO llegué al límite de epochs
while (((math.fabs(error_epoch - error_ant) > error_delta_umbral) or( error_epoch>0.01) ) and (epoch < epoch_limit) ):
    epoch += 1
    error_suma = 0.0
    error_ant = error_epoch

    # recorro siempre TODOS los registros de entrada
    orden = np.random.permutation(X_row)
    for fila in orden:
        # fila es el registro actual
        x = X[fila:fila+1,:] # ej  array([[-1, -1]])
        clase = Y[fila:fila+1,:]  # shape (1, Y_col), para que clase.T sea (Y_col, 1)

        # propagar el x hacia adelante, FORWARD
        entrada = x.T  # la entrada a la red
        niveles = red["arq"]["layers_qty"]

        # etapa forward
        # recorro hacia adelante, nivel a nivel
        vsalida =  [0] *(niveles) # salida de cada nivel de la red

        for i in range(niveles):
          estimulos = red["layer"][i]["W"] @ entrada + red["layer"][i]["w0"]
          vsalida[i] =  func_eval_vec(red["layer"][i]["func"], estimulos)
          entrada = vsalida[i]  # para la proxima vuelta


        # etapa backward
        # calculo los errores en la capa hidden y la capa output
        verror =  [0] *(niveles+1) # inicializo dummy
        verror[niveles] = clase.T - vsalida[niveles-1]

        i = niveles-1
        verror[i] = verror[i+1] * deriv_eval_vec(red["layer"][i]["func"], vsalida[i])

        for i in reversed(range(niveles-1)):
           verror[i] = deriv_eval_vec(red["layer"][i]["func"], vsalida[i])*(red["layer"][i+1]["W"].T @ verror[i+1])

        # ya tengo los errores que comete cada capa
        # corregir matrices de pesos, voy hacia atras
        # backpropagation
        entrada = x.T
        for i in range(niveles):
          red["layer"][i]["W"]  =  red["layer"][i]["W"] + learning_rate *(verror[i] @ entrada.T)
          red["layer"][i]["w0"] =  red["layer"][i]["w0"] + learning_rate * verror[i]
          entrada = vsalida[i]  # para la proxima vuelta



    # ya recalcule las matrices de pesos
    # ahora avanzo la red, feed-forward
    # para calcular el red(X) = Y
    entrada = X.T
    for i in range(niveles):
        estimulos = red["layer"][i]["W"] @ entrada + red["layer"][i]["w0"]
        salida =  func_eval_vec(red["layer"][i]["func"], estimulos)
        entrada = salida  # para la proxima vuelta

    output_salidas = salida  # salida tiene la salida del ultimo layer

    # calculo el error promedio general de TODOS los X
    error_epoch= np.mean( (Y.T - output_salidas)**2 )

    # log de avance: primer epoch, cada 'log_cada' epochs, o el ultimo
    if epoch == 1 or epoch % log_cada == 0 or epoch == epoch_limit:
        delta = math.fabs(error_epoch - error_ant)
        print(f"{epoch:>6} | {error_epoch:>12.6e} | {delta:>12.6e}")


print("-" * 40)
print(f"Entrenamiento finalizado en epoch {epoch}")

# el error
print("error_epoch= ", error_epoch)
print("error_ant = ", error_ant)
print("delta = ", math.fabs(error_epoch - error_ant))


# la cantidad de epochs necesarias hasta encontrar una solucion
print("Para converger hicieron falta epochs=",epoch)


# imprimo los niveles de la red
#for i in range(red["arq"]["layers_qty"]):
#  print( red["layer"][i])


# calculo la salida de la red
#  comprouebo que NO son valores 0 o 1
#  lo que implica que deberé decidir mediante un umbral

niveles = red["arq"]["layers_qty"]

print(f"fila\treal\t\tpred\t\tconfianza")
aciertos = 0
for fila in range(X_row):
    # fila es el registro actual
    x = X[fila:fila+1,:]
    real = clases_str[fila]
    entrada = x.T  # la entrada a la red

    # etapa forward
    # recorro hacia adelante, nivel a nivel
    for i in range(niveles):
       estimulos = red["layer"][i]["W"] @ entrada + red["layer"][i]["w0"]
       salida =  func_eval_vec(red["layer"][i]["func"], estimulos)
       entrada = salida  # para la proxima vuelta

    pred_idx = int(np.argmax(salida))
    pred = clases_unicas[pred_idx]
    conf = float(salida[pred_idx, 0])
    ok = (pred == real)
    if ok: aciertos += 1
    #print(f"{fila}\t{real:12s}\t{pred:12s}\t{conf:.4f}\t{'ok' if ok else 'FAIL'}")

print(f"\nAccuracy training: {aciertos}/{X_row} = {aciertos/X_row*100:.2f}%")

with open(TEST_TXT, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header_t = next(reader)
    filas_t = list(reader)
y_idx_t = header_t.index("y")
x_idxs_t = [i for i in range(len(header_t)) if i != y_idx_t]
X_test = np.array([[float(r[i]) for i in x_idxs_t] for r in filas_t], dtype=np.float64)
y_test = np.array([r[y_idx_t] for r in filas_t])
# uso el MISMO scaler que entrené con training (no fit, sólo transform)
X_test = scaler.transform(X_test)
print("\n========= TEST =========")
print(f"fila\treal\t\tpred\t\tconfianza")
aciertos_test = 0
for fila in range(X_test.shape[0]):
    x = X_test[fila:fila+1, :]
    real = y_test[fila]
    entrada = x.T
    for i in range(niveles):
        estimulos = red["layer"][i]["W"] @ entrada + red["layer"][i]["w0"]
        salida = func_eval_vec(red["layer"][i]["func"], estimulos)
        entrada = salida
    pred_idx = int(np.argmax(salida))
    pred = clases_unicas[pred_idx]
    conf = float(salida[pred_idx, 0])
    ok = (pred == real)
    if ok:
        aciertos_test += 1
    print(f"{fila}\t{real:12s}\t{pred:12s}\t{conf:.4f}\t{'ok' if ok else 'FAIL'}")
n_test = X_test.shape[0]
print(f"\nAccuracy test: {aciertos_test}/{n_test} = {aciertos_test/n_test*100:.2f}%")


# ===========================
# Evaluación sobre TEST con UMBRAL de confianza
# ===========================

UMBRAL_CONFIANZA = 0.90
print(f"\n========= TEST con umbral de confianza >= {UMBRAL_CONFIANZA} =========")
print(f"fila\treal\t\tpred\t\tconfianza\tresultado")

aciertos_umbral = 0
errores_umbral = 0
descartados = 0

for fila in range(X_test.shape[0]):
    x = X_test[fila:fila+1, :]
    real = y_test[fila]
    entrada = x.T

    for i in range(niveles):
        estimulos = red["layer"][i]["W"] @ entrada + red["layer"][i]["w0"]
        salida = func_eval_vec(red["layer"][i]["func"], estimulos)
        entrada = salida

    pred_idx = int(np.argmax(salida))
    pred = clases_unicas[pred_idx]
    conf = float(salida[pred_idx, 0])

    if conf < UMBRAL_CONFIANZA:
        descartados += 1
        resultado = "DESCARTADO"
    elif pred == real:
        aciertos_umbral += 1
        resultado = "ok"
    else:
        errores_umbral += 1
        resultado = "FAIL"

    print(f"{fila}\t{real:12s}\t{pred:12s}\t{conf:.4f}\t{resultado}")

aceptados = aciertos_umbral + errores_umbral
print("\n--- Resumen test con umbral ---")
print(f"Total registros:  {n_test}")
print(f"Descartados:      {descartados}/{n_test} ({descartados/n_test*100:.2f}%)")
print(f"Aceptados:        {aceptados}/{n_test} ({aceptados/n_test*100:.2f}%)")
if aceptados > 0:
    print(f"  Aciertos:       {aciertos_umbral}/{aceptados} ({aciertos_umbral/aceptados*100:.2f}%)")
    print(f"  Errores:        {errores_umbral}/{aceptados} ({errores_umbral/aceptados*100:.2f}%)")


# ===========================
# Evaluación sobre TEST CUSTOM (original_custom_test/ -> caras_1200/custom_test/)
# ===========================

if not TEST_CUSTOM_TXT.exists():
    print(f"\n========= TEST CUSTOM =========")
    print(f"No existe {TEST_CUSTOM_TXT}; salteo evaluacion custom.")
    print("(Dropea fotos en original_custom_test/<persona>/ y corre procesar.py para generarlo.)")
else:
    with open(TEST_CUSTOM_TXT, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header_c = next(reader)
        filas_c = list(reader)

    if not filas_c:
        print(f"\n========= TEST CUSTOM =========")
        print(f"{TEST_CUSTOM_TXT} esta vacio (solo header); salteo evaluacion custom.")
    else:
        y_idx_c = header_c.index("y")
        x_idxs_c = [i for i in range(len(header_c)) if i != y_idx_c]
        X_custom = np.array([[float(r[i]) for i in x_idxs_c] for r in filas_c], dtype=np.float64)
        y_custom = np.array([r[y_idx_c] for r in filas_c])
        # uso el MISMO scaler que entrene con training
        X_custom = scaler.transform(X_custom)
        n_custom = X_custom.shape[0]

        print(f"\n========= TEST CUSTOM con umbral de confianza >= {UMBRAL_CONFIANZA} =========")
        print(f"fila\treal\t\tpred\t\tconfianza\tresultado")
        aciertos_cu_umbral = 0
        errores_cu_umbral = 0
        descartados_cu = 0
        for fila in range(n_custom):
            x = X_custom[fila:fila+1, :]
            real = y_custom[fila]
            entrada = x.T
            for i in range(niveles):
                estimulos = red["layer"][i]["W"] @ entrada + red["layer"][i]["w0"]
                salida = func_eval_vec(red["layer"][i]["func"], estimulos)
                entrada = salida
            pred_idx = int(np.argmax(salida))
            pred = clases_unicas[pred_idx]
            conf = float(salida[pred_idx, 0])
            if conf < UMBRAL_CONFIANZA:
                descartados_cu += 1
                resultado = "DESCARTADO"
            elif pred == real:
                aciertos_cu_umbral += 1
                resultado = "ok"
            else:
                errores_cu_umbral += 1
                resultado = "FAIL"
            print(f"{fila}\t{real:12s}\t{pred:12s}\t{conf:.4f}\t{resultado}")

        aceptados_cu = aciertos_cu_umbral + errores_cu_umbral
        print("\n--- Resumen test custom con umbral ---")
        print(f"Total registros:  {n_custom}")
        print(f"Descartados:      {descartados_cu}/{n_custom} ({descartados_cu/n_custom*100:.2f}%)")
        print(f"Aceptados:        {aceptados_cu}/{n_custom} ({aceptados_cu/n_custom*100:.2f}%)")
        if aceptados_cu > 0:
            print(f"  Aciertos:       {aciertos_cu_umbral}/{aceptados_cu} ({aciertos_cu_umbral/aceptados_cu*100:.2f}%)")
            print(f"  Errores:        {errores_cu_umbral}/{aceptados_cu} ({errores_cu_umbral/aceptados_cu*100:.2f}%)")
