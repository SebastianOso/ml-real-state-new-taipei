# ml-real-state-new-taipei

Implementación de un modelo de regresión lineal múltiple desde cero para predecir el precio por unidad de área de viviendas en Nuevo Taipei, Taiwán.
 
Para una explicación completa de la metodología, el análisis exploratorio, la construcción del modelo y los resultados, consultar el paper:
 
`Predicción_del_Precio_de_Viviendas_mediante_Regresión_Lineal_Múltiple_usando_Machine_Learning.pdf`

## Requisitos
 
- Python 3.8 o superior
- pip
---
 
## Configuración del entorno
 
### 1. Clonar el repositorio
 
```bash
git clone https://github.com/SebastianOso/ml-real-state-new-taipei.git
cd ml-real-state-new-taipei
```
 
### 2. Crear el entorno virtual
 
```bash
python3 -m venv venv
```
 
### 3. Activar el entorno virtual
 
**En macOS / Linux:**
```bash
source venv/bin/activate
```
 
**En Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```
 
**En Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```
 
> Sabrás que está activado porque el prompt cambia y muestra `(venv)` al inicio.
 
### 4. Instalar dependencias
 
```bash
pip install -r requirements.txt
```
 
### 5. Correr el proyecto
 
```bash
cd notebooks
python linear_regression_real_state.py
```
 
---
 
## Desactivar el entorno virtual
 
Cuando termines de trabajar, desactiva el venv con:
 
```bash
deactivate
```
 
---
 