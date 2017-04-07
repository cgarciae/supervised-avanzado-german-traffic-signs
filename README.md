# Reto German Traffic Signs
## Descripcion

## Getting Started
Para resolver este reto primero has un fork de este repositorio, y clona tu fork.

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

# Python
