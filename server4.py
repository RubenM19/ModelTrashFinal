from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import uvicorn
import joblib
from pydantic import BaseModel

app_desc = """<h2>Subida de Imagenes `predict/image`</h2>
<h2>BASURA DETECTION</h2>"""

app = FastAPI(title='PROYECTO BASURITA', description=app_desc)

# Variables globales para el modelo de imagen, el modelo Random Forest y el scaler
image_model = None
svc_model = None
scaler = None

# Carga del modelo de imagen
try:
    image_model = load_model("./models/model3.keras")
    input_shape = image_model.input_shape[1:3]  # Obtener el tamaño de entrada del modelo (altura y ancho)
    print(f"Modelo de imagen cargado correctamente. Tamaño de entrada esperado: {input_shape}")
except Exception as e:
    print(f"Error al cargar el modelo de imagen: {e}")

# Carga del modelo de SVC y el scaler usando joblib
try:
    svc_model = joblib.load('./models/SVC_model_Pred.pkl')
    print("Modelo SVC cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo SVC: {e}")
    svc_model = None

try:
    scaler = joblib.load('./models/SVC_scaler_Pred.pkl')
    print("Scaler cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el scaler: {e}")
    scaler = None

def prepare_image(img: Image.Image, target_size: tuple) -> np.ndarray:
    """Prepara la imagen para la predicción."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalización
    return img_array

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Verifica que el archivo sea una imagen
    print(file.content_type)
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")
    try:
        # Lee la imagen
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Prepara la imagen
        img_array = prepare_image(img, target_size=input_shape)  # Usar el tamaño de entrada del modelo

        # Realiza la predicción
        prediction = image_model.predict(img_array)
        predicted_class = "trash" if prediction[0][0] > 0.5 else "clean"
        print(prediction)
        print(prediction[0][0])
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {e}")

class PredictValues(BaseModel):
    Heart_Rate: float
    Respiracion: float
    SO2: float
    Temperature: float

@app.post("/predict/values")
async def predict_values(values: PredictValues):
    try:
        if scaler is None:
            raise ValueError("Scaler no está cargado correctamente.")
        if svc_model is None:
            raise ValueError("Modelo SVC no está cargado correctamente.")
        
        # Normalizar los datos de entrada
        input_data = np.array([[values.Heart_Rate, values.Respiracion, values.SO2, values.Temperature]])
        scaled_input = scaler.transform(input_data)

        # Realizar la predicción
        prediction = svc_model.predict(scaled_input)
        result = 'Negative' if prediction[0] == 1 else 'Positive'
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando los valores: {e}")

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
