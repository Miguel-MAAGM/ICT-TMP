import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CONFIGURACIÓN DE FILTROS Y PARÁMETROS
# ==========================================
# Nombre de tu archivo original
archivo_entrada = 'test_1.csv'  

# DEFINE AQUÍ TUS LÍMITES PARA EL EJE Y
# Modifica estos números según la sección que quieras recortar
Y_MIN = -250.0   
Y_MAX = 250.0    

# ==========================================
# 2. CARGA Y FILTRADO DE DATOS
# ==========================================
print("Cargando datos...")
df = pd.read_csv(archivo_entrada)

print(f"Puntos totales originales: {len(df)}")

# Aplicamos la restricción en el eje Y
df_filtrado = df[(df['y'] >= Y_MIN) & (df['y'] <= Y_MAX)]

print(f"Puntos tras restringir Y (entre {Y_MIN} y {Y_MAX}): {len(df_filtrado)}")

# ==========================================
# 3. VISUALIZACIÓN INTERACTIVA 3D
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dibujamos los puntos filtados
# 'c=df_filtrado['z']' pinta los puntos con un degradado según su altura
# 's=1' define el tamaño del punto (puedes subirlo a 2 o 3 si se ven muy pequeños)
mapa_color = ax.scatter(
    df_filtrado['x'], 
    df_filtrado['y'], 
    df_filtrado['z'], 
    c=df_filtrado['z'], 
    cmap='viridis', 
    s=1
)

# Configuración de etiquetas de los ejes
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y (Restringido)')
ax.set_zlabel('Eje Z')
ax.set_title(f'Nube de Puntos 3D - Filtro Y: [{Y_MIN}, {Y_MAX}]')

# Añadir una barra de color lateral que indica la altura (Z)
barra_color = fig.colorbar(mapa_color, ax=ax, pad=0.1)
barra_color.set_label('Altura (Z)')

# Ajustar los límites visuales del gráfico para que coincidan con el filtro
ax.set_ylim(Y_MIN, Y_MAX)

print("Abriendo visor 3D... (Puedes rotar el gráfico manteniendo el click izquierdo)")
plt.show()