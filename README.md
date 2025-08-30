# Numpy_Pandas
Comandos principales de NumPy y Pandas


# B) NumPy – Operaciones estadísticas y funciones avanzadas

NumPy incluye un conjunto de funciones matemáticas y estadísticas que permiten analizar y transformar arreglos de forma eficiente.  

---

### 1. Estadísticos básicos
- **`np.mean()`** → devuelve el promedio de los valores.  
- **`np.std()`** → calcula la desviación estándar (variabilidad de los datos).  
- **`np.sum()`** → suma todos los elementos.  
- Estas funciones admiten el parámetro `axis` para operar por filas o columnas.  
  - Ejemplo: `np.sum(arr, axis=0)` → suma por columnas.  

---

### 2. Generación de secuencias numéricas
- **`np.arange(inicio, fin, paso)`** → genera una secuencia con un intervalo definido.  
  - Ejemplo: `np.arange(0,10,2)` → `[0,2,4,6,8]`.  
- **`np.linspace(inicio, fin, num)`** → genera una secuencia con `num` puntos equidistantes.  
  - Ejemplo: `np.linspace(0,1,5)` → `[0.,0.25,0.5,0.75,1.]`.  

---

### 3. Aleatoriedad y simulaciones
- **`np.random.rand()`** → números aleatorios en distribución uniforme [0,1).  
- **`np.random.randn()`** → números aleatorios en distribución normal estándar.  
- **`np.random.seed(valor)`** → asegura reproducibilidad (los mismos resultados cada vez).  

---

### 4. Álgebra lineal
NumPy incluye un submódulo para resolver problemas de álgebra lineal:  
- **`np.linalg.det(A)`** → determinante de la matriz `A`.  
- **`np.linalg.inv(A)`** → inversa de la matriz `A`.  
- **`np.linalg.solve(A, b)`** → resuelve sistemas de ecuaciones lineales `Ax=b`.  

**Ejemplo:**  
Si `A = [[3,2],[1,2]]` y `b = [2,0]`, la solución es `x = [1, -0.5]`.  

---
=======
### A) NumPy – Arrays y Reshape
import numpy as np

## Crear un arreglo 1D
a = np.array([1, 2, 3, 4])
print("Arreglo 1D:", a)
print("Forma:", a.shape)
print("Número de dimensiones:", a.ndim)

## Crear un arreglo 2D
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print("\nArreglo 2D:\n", b)
print("Forma:", b.shape)
print("Número de dimensiones:", b.ndim)

## Usar arange y reshape
c = np.arange(12)      # números 0 a 11
print("\nArreglo creado con arange:", c)

c2 = c.reshape(3, 4)   # redimensionar a 3x4
print("\nArreglo reshape (3x4):\n", c2)

c3 = c.reshape(2, 2, 3) # redimensionar a 3D
print("\nArreglo reshape (2x2x3):\n", c3)


### Arreglo 1D: [1 2 3 4]

Forma: (4,)

Número de dimensiones: 1

### Arreglo 2D:
 
 [[1 2 3]
 
  [4 5 6]]
  
Forma: (2, 3)
Número de dimensiones: 2

Arreglo creado con arange: [ 0  1  2  3  4  5  6  7  8  9 10 11]

### Arreglo reshape (3x4):

 [[ 0  1  2  3]
 
  [ 4  5  6  7]
  
  [ 8  9 10 11]]

### Arreglo reshape (2x2x3):

 [[[ 0  1  2]
 
   [ 3  4  5]]

  [[ 6  7  8]
  
   [ 9 10 11]]]



### A) NumPy – Concatenación y operaciones básicas

import numpy as np

## Concatenación horizontal y vertical
x = np.array

              ([[1, 2],
              [3, 4]])
y = np.array
    
              ([[5, 6],
              [7, 8]])

print("x:\n", x)

print("y:\n", y)

### cat0 = np.concatenate

([x, y], axis=0) # filas

print("\nConcatenación eje 0 (vertical):\n", cat0)

## cat1 = np.concatenate([x, y], axis=1)  # columnas

print("\nConcatenación eje 1 (horizontal):\n", cat1)

## Operaciones básicas y broadcasting
u = np.array  

                ([10, 20, 30, 40])
v = np.array  

                ([1, 2, 3, 4])

print("\nu + v:", u + v)
print("u * v:", u * v)
print("u * 2:", u * 2)

mat = np.array

                ([[1, 2, 3, 4],
                [1, 2, 3, 4]])
                
print("\nmat + u (broadcasting):\n", mat + u)





=======
# C) Pandas – Creación y manipulación de DataFrames y Series

C-pandas-dataframes
## 1. Creación de Series
Un `Series` es una columna unidimensional (como un vector con etiquetas).
```python

import pandas as pd

# Crear un Series a partir de una lista
serie = pd.Series([10, 20, 30, 40], name="Edades")

print("Series:")
display(serie)

```


## 2. Creación de DataFrames
Un DataFrame es la estructura principal de Pandas.
Piensa en él como una tabla de datos con filas y columnas (similar a una hoja de cálculo o una tabla SQL).
Está compuesto por varias Series organizadas en columnas.
```python
# Crear un DataFrame a partir de un diccionario
data = {
    "Nombre": ["Ana", "Luis", "Carlos", "María"],
    "Edad": [23, 30, 27, 22],
    "Ciudad": ["Bogotá", "Medellín", "Cali", "Barranquilla"]
}

df = pd.DataFrame(data)

print("DataFrame:")
display(df)
=======
```

## 4. Selección de columnas
```python
# Seleccionar una sola columna
print("Columna Edad:")
display(df["Edad"])

# Seleccionar múltiples columnas
print("Columnas Nombre y Ciudad:")
display(df[["Nombre", "Ciudad"]])
```

## D) Pandas – Operaciones avanzadas en DataFrames
# 1. Filtros

Sirven para seleccionar filas según una condición.
Ejemplo:
```python
import pandas as pd

df = pd.DataFrame({
    "Nombre": ["Ana", "Luis", "Carlos", "María"],
    "Edad": [23, 34, 29, 40],
    "Ciudad": ["Bogotá", "Medellín", "Cali", "Bogotá"]
})

# Filtrar personas mayores de 30
filtro = df[df["Edad"] > 30]
print(filtro)

# Filtrar por ciudad
print(df[df["Ciudad"] == "Bogotá"])