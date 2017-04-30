# Reto German Traffic Signs
## Descripcion
El [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) es un conjunto de imágenes de señales de tránsito Alemanas.

![alt text][s1] ![alt text][s2] ![alt text][s3] ![alt text][s4] ![alt text][s5] ![alt text][s6] ![alt text][s7] ![alt text][s8] ![alt text][s9]

Este dataset tiene mas de 50,000 imágenes separadas en 43 clases. El reto es construir un clasificador de imágenes que sea capaz de reconocer estas señales de tránsito. Adicionalmente, el dataset incluye las posiciones (aka bounding boxes) de los objetos dentro de la imagen.

### Formato Datos
Todos los datos viven en la carpeta `.dataget/data` y se dividen en 2 grupos
```
|- .dataget
   |- data
      |- german-traffic-signs
         |- traning-set
         |- test-set
```
**Estructura** <br>
Cada sub conjunto de datos (training-set o test-set) tiene la siguiente estructura
```
|- {subset}
   |- {class_id}
      |- {class_id}.csv
      |- {image_id}.jpg
```
Donde
* `subset`: unicamente puede ser `training-set` o `test-set`
* `class_id`: es el identificador de una clase, e.g. 0, 1, 2... 42
* `image_id`: es el nombre de una imagen.

Por ejemplo
```
|- training-set
   |- 0
      |- 0.csv
      |- 000000.jpg
      |- 000001.jpg
      |- 000002.jpg
      ...
   |- 1
      |- 1.csv
      |- 000000.jpg
      |- 000001.jpg
      |- 000002.jpg
      ...
```


### Variables
Todos los archivos `*.csv` contienen las siguiente variables

| filename | width | height | roi.x1,  roi.y1,  roi.x2, roi.y2 | class_id |
| - |  - |  - |  - |  - |
| Archivo de la imagen a la que corresponde esta informacion | Ancho de la imagen | Alto de la imagen | Informacion del bounding box | Numero entero que indica la clase a la que pretenece la imagen |

Cada imagen como tal puede ser representada por una matriz 3D de dimensiones `height x width x 3` dado que es RGB. Se recomienda redimensionar cada imagen a `32 x 32 x 3`.


### Objetivo
1. Crear un algoritmo que tome una imagen de entrada, ya sea como vector o matriz, y retorne el clase (`class_id`) a la que pertenece esa imagen.
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

### Descarga y Preprocesamiento
Para descargar los datos ejecuta el comando
```bash
dataget get german-traffic-signs
```
Esto descarga los archivos en la carpeta `.dataget/data`, los divide en los conjuntos `training-set` y `test-set`, convierte las imagenes en `jpg` de dimensiones `32x32`. Las originalmente vienen en formato `.ppm` y con dimensiones variables. Si deseas mantener el formato original ejecuta en vez

```bash
dataget get --dont-process german-traffic-signs
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
