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
**Training Set** <br>
La carpeta `data/traning-set` esta organizada de la siguiente manera
* En `data/training-set/GTSRB/Final_Training/Images` se encuentran 43 carpeta, desde la carpeta `000000` que hasta la carpeta `000042`, cada una de las carpetas contiene todas las imagenes de una clase.
* Las imagenes originalmente vienen en el formato `.ppm` y tienen dimensiones variables (ver [descarga](#descarga)), sin embargo, si ejecutas el script `preprocess_data.py` este las transforma a `.jpg` y las redimensiona a `32x32`.  (ver [preprocesamiento](#preprocesamiento)).
* Dentro de la carpeta de cada clase existe un archivo de la forma `GT-{folder_clase}.csv` que contiene datos de las imagenes en esa clase.
* El archivo `data/training-set/GTSRB/Readme-Images.txt` contiene informacion adicional sobre el dataset.

**Test Set** <br>
La carpeta `data/traning-set` esta organizada de la siguiente manera
* En `data/test-set/GTSRB/Final_Test/Images` todas las imagenes del test-set.
* Las imagenes originalmente vienen en el formato `.ppm` y tienen dimensiones variables (ver [descarga](#descarga)), sin embargo, si ejecutas el script `preprocess_data.py` este las transforma a `.jpg` y las redimensiona a `32x32`.  (ver [preprocesamiento](#preprocesamiento)).
* Dentro de la carpeta de imagenes existe el archivo `GT-final_test.csv` que contiene los datos de las imagenes.
* El archivo `data/test-set/GTSRB/Readme-Images-Final-test.txt` contiene informacion adicional sobre el dataset.

### Variables
Todos los archivos `*.csv` contienen las siguiente variables
| Filename | Width | Height | Roi.X1,  Roi.Y1,  Roi.X2, Roi.Y2 | ClassId |
| - |  - |  - |  - |  - |
| Archivo de la imagen es que corresponde esta informacion | Ancho de la imagen | Alto de la imagen | Informacion del bounding box | Numero entero que indica la clase a la que pretenece la imagen |

Cada imagen como tal puede ser representada por una matriz 3D de dimensiones `Height x Width x 3` dado que es RGB. Se recomienda redimensionar cada imagen a `32 x 32 x 3`, el script `preprocess_data.py` realiza esta operacion sobre los datos en disco.

*Nota: el script `preprocess_data.py` altera las dimensiones de la imagen y los datos de los `.csv` por ahora no son modificados acorde.*

### Objetivo
1. Crear un algoritmo que tome una imagen de entrada, ya sea como vector o matriz, y retorne el clase (`ClassId`) a la que pertenece esa imagen.
1. Entrenar este algoritmo utilizando los datos de la carpeta `data/training-set`.
1. Medir el performance/score del algoritmo utilizando los datos de la carpeta `data/test-set`. El performance debe ser medido como
```python
score = n_aciertos / n_imagenes * 100
```
donde `n_aciertos` es el numero de imagenes clasificadas de forma correcta y `n_imagenes` es el numero total de imagenes en el `test-set`.

### Notas Teoricas
* Dado que las imagenes son conjuntos con dimensiones muy altas, usualmente la mejor manera de atacar el problema es utilizando [redes neuronales](https://en.wikipedia.org/wiki/Artificial_neural_network).
  * Para imagenes es recomendable utilizar redes [convolucionales](http://cs231n.github.io/convolutional-networks/).

### Solucion
Ver procedimiento de [solucion](https://github.com/colomb-ia/formato-retos#solucion).

##### Requerimientos
*Indica los requerimientos para utilizar el codigo de tu solucion.*

##### Procedimiento
*Indica el procedimiento que se debe seguir para reproducir tu solucion.*

##### Metodo
*Indica el metodo que utilizaste para solucionar el reto.*

##### Resultados
*Indica el metodo que utilizaste para solucionar el reto.*

## Getting Started
Para resolver este reto primero has un [fork](https://help.github.com/articles/fork-a-repo/) de este repositorio y [clona](https://help.github.com/articles/cloning-a-repository/) el fork en tu maquina.

```bash
git clone https://github.com/{username}/supervised-avanzado-german-traffic-signs
cd supervised-avanzado-german-traffic-signs
```

*Nota: reemplaza `{username}` con tu nombre de usuario de Github.*

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
