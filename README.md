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

