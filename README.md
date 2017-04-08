# Reto German Traffic Signs
## Descripcion
El [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) es un conjunto de imágenes de señales de tránsito Alemanas.

![alt text][s1] ![alt text][s2] ![alt text][s3] ![alt text][s4] ![alt text][s5] ![alt text][s6] ![alt text][s7] ![alt text][s8] ![alt text][s9]

Este dataset tiene mas de 50,000 imágenes separadas en 43 clases. El reto es construir un clasificador de imágenes que sea capaz de reconocer estas señales de tránsito. Adicionalmente, el dataset incluye las posiciones (aka bounding boxes) de los objetos dentro de la imagen.

### Formato Datos
Todos los datos viven en la carpeta `data` y se dividen en 2 grupos
```
|-data
| |-traning-set
| |-test-set
```


### Variables
*Describe cada una de las variables de los datos*

### Objetivo
*Especifica el objetivo del reto y el rol de las variables (e.g cuales son los features y cuales son los labels). IMPORTANTE: se debe especificar que el performance debe ser calculado con respecto a los datos del `test-set` y este no puede ser utilizado para el entrenamiento.*

### Solucion
Ver procedimiento de [solucion](https://github.com/colomb-ia/formato-retos#solucion).

##### Requerimientos
*Indica los requerimientos para utilizar el codigo de tu solucion*

##### Procedimiento
*Indica el procedimiento que se debe seguir para reproducir tu solucion*

##### Metodo
*Indica el metodo que utilizaste para solucionar el reto*

##### Resultados
*Indica el metodo que utilizaste para solucionar el reto*

### Notas Teoricas
*Sugiere algunos aspectos teoricos a tener en cuenta.*

## Getting Started
Para resolver este reto primero has un [fork](https://help.github.com/articles/fork-a-repo/) de este repositorio y [clona](https://help.github.com/articles/cloning-a-repository/) el fork en tu maquina.

```bash
git clone https://github.com/{username}/supervised-avanzado-german-traffic-signs
cd supervised-avanzado-german-traffic-signs
```

### Requerimientos
Para descargar y visualizar los datos necesitas Python 2 o 3. Las dependencias las puedes encontrar en el archivo `requirements.txt`, el cual incluye
* pillow
* numpy
* pandas
* jupyter

Puedes instalarlas fácilmente utilizando el commando

```bash
pip install -r requirements.txt
```
Dependiendo de tu entorno puede que necesites instalar paquetes del sistema adicionales, si tienes problemas revisa la documentación de estas librerías.

### Descarga
Para descargar los datos ejecuta el comando
```bash
python download_data.py
```
Esto descarga los archivos en la carpeta `data`. Los datos se divide en 2 conjuntos: `training-set` y `test-set`, cada conjunto vive dentro de su propia carpeta.

### Preprocesamiento
Las imágenes del formato original es `.ppm` y las dimensiones de estas varían. Si deseas convertirlas a `.jpg` y redimensionarlas a `32x32` ejecuta

```bash
python process_data.py
```

# Starter Code Python
Para iniciar con este reto puedes correr el codigo de Python en Jupyter del archivo `python-sample.ipynb`. Este código que ayudará a cargar y visualizar algunas imágenes. Las dependencias son las mismas que se instalaron durante la descarga de los datos, ver [Requerimientos](#requerimientos).

Para iniciar el código solo hay que prender Jupyter en esta carpeta

```bash
jupyter notebook .
```
y abrir el archivo `python-sample.ipynb`.


# Soluciones
| Score | Usuario |	Algoritmo | Link Repo |
| - | - | - | - |
| *score* | *nombre* | *algoritmo* | *link* |



[s1]: http://benchmark.ini.rub.de/Images/gtsrb/0.png "S"
[s2]: http://benchmark.ini.rub.de/Images/gtsrb/1.png "S"
[s3]: http://benchmark.ini.rub.de/Images/gtsrb/2.png "S"
[s4]: http://benchmark.ini.rub.de/Images/gtsrb/3.png "S"
[s5]: http://benchmark.ini.rub.de/Images/gtsrb/4.png "S"
[s6]: http://benchmark.ini.rub.de/Images/gtsrb/5.png "S"
[s7]: http://benchmark.ini.rub.de/Images/gtsrb/6.png "S"
[s8]: http://benchmark.ini.rub.de/Images/gtsrb/11.png "S"
[s9]: http://benchmark.ini.rub.de/Images/gtsrb/8.png "S"
