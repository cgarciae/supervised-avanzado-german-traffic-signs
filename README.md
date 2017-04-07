# Reto German Traffic Signs
## Descripcion
El [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) es un conjunto de imagenes de señales de transito Alemanas.

![alt text][s1] ![alt text][s2] ![alt text][s3] ![alt text][s4] ![alt text][s5] ![alt text][s6] ![alt text][s7] ![alt text][s8] ![alt text][s9]

Este dataset tiene mas de 50,000 imagenes separadas en 43 clases. El reto es construir un clasificador de imagenes que sea capaz de reconocer estas señales de transito. Adicionalmente, el dataset incluye las posicion (aka bounding boxes) de los objetos dentro de la imagen.

## Getting Started
Para resolver este reto primero has un fork de este repositorio y clonalo en tu maquina.

### Requerimientos
Para descargar y visualizar los datos necesitas Python 2 o 3. Las dependencias las puedes encontrar en el archivo `requirements.txt`, el cual incluye
* pillow
* numpy
* pandas
* jupyter

Puedes instalarlas facilmente utilizando el commando

```bash
pip install -r requirements.txt
```
Dependiendo de tu enterno puede que necesites instalar paquetes del sistema adicionales, si tienes problemas revisa la documentacion de estas librerias.

### Descarga
Para descargar los datos ejecuta el comando
```bash
python download_data.py
```
Esto descarga los archivos en la carpeta `data`.

### Preprocesamiento
Las imagenes del formato original es `.ppm` y las dimensiones de estas varian. Si deseas convertirlas a `.jpg` y redimensionarlas a `32x32` ejecuta

```bash
python process_data.py
```

# Starter Code Python
Para iniciar con este reto puedes correr el codigo de Python en Jupyter del archivo `python-sample.ipynb`. Este codigo que ayudara a cargar y visualizar algunas imagenes. Las dependencias son las mismas que se instalaron durante la descarga de los datos, ver [Requerimientos](#Requerimientos).

Para iniciar el codigo solo hay que prender Jupyter en esta carpeta

```bash
jupyter notebook .
```
y abrir el archivo `python-sample.ipynb`.

[s1]: http://benchmark.ini.rub.de/Images/gtsrb/0.png "S"
[s2]: http://benchmark.ini.rub.de/Images/gtsrb/1.png "S"
[s3]: http://benchmark.ini.rub.de/Images/gtsrb/2.png "S"
[s4]: http://benchmark.ini.rub.de/Images/gtsrb/3.png "S"
[s5]: http://benchmark.ini.rub.de/Images/gtsrb/4.png "S"
[s6]: http://benchmark.ini.rub.de/Images/gtsrb/5.png "S"
[s7]: http://benchmark.ini.rub.de/Images/gtsrb/6.png "S"
[s8]: http://benchmark.ini.rub.de/Images/gtsrb/11.png "S"
[s9]: http://benchmark.ini.rub.de/Images/gtsrb/8.png "S"
