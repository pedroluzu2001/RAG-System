# **RedTec RAG Project**

## **Descripción**
Proyecto de **RAG (Retrieval-Augmented Generation)** usando **FastAPI**, **FAISS** y **GPT-4o-mini** vía OpenRouter.  
Permite hacer preguntas sobre documentos cargados en la carpeta `redtec_docs` y obtener respuestas generadas por el modelo LLM basadas en contexto.

<img width="595" height="300" alt="image" src="https://github.com/user-attachments/assets/f6d7d3b9-22c3-4232-9e68-1de39ee2522e" />


## **Requisitos**

- **Python:** 3.10 o superior  
- **Gestor de paquetes:** pip  
- **API Key:** OpenRouter API Key  
- **Sistema operativo compatible:** Windows, Linux o macOS  
- **Dependencias adicionales:** Listadas en `requirements.txt`


## **Instalación**

1. **Clonar el repositorio**

```bash
git clone https://github.com/pedroluzu2001/RAG-System.git
cd redtech-rag-project
```
**Linux / Mac**
```
python3 -m venv rag-env
source rag-env/bin/activate
```
**Instalar dependencias**
```
pip install -r requirements.txt
```

**Configurar tu API key de OpenRouter**

Windows (PowerShell)
```
$env:OPENROUTER_API_KEY="TU_API_KEY_AQUI"
```

Linux / Mac
```
export OPENROUTER_API_KEY="TU_API_KEY_AQUI"
```
Correr el servidor
```
python main.py
```

La API estará disponible en: http://127.0.0.1:8000

Swagger UI para probar endpoints: http://127.0.0.1:8000/docs

**Uso:**
```
Endpoint /ask

Método: POST

Content-Type: application/json

Ejemplo de petición
{
  "question": "¿Qué se habló en la reunión de infraestructura?",
  "top_k": 3
}

Ejemplo de respuesta
{
  "question": "¿Qué se habló en la reunión de infraestructura?",
  "answer": "Se discutió la migración parcial de entrenamiento de modelos a AWS, comparación de costos entre AWS Sagemaker y Azure ML, y configurar VPN para acceso seguro al clúster híbrido.",
  "retrieved_contexts": [
    {
      "source": "reunion_infraestructura_cloud.txt",
      "chunk": 0,
      "score": 1.87,
      "text": "Fecha: 2 de marzo de 2025\nTema: Infraestructura de cómputo en la nube..."
    }
  ]
}
```

Nota: retrieved_contexts muestra los fragmentos de documentos utilizados por el modelo para generar la respuesta.

**Agregar más documentos**

1._Coloca archivos .txt, .pdf o .md en la carpeta redtec_docs/.

2._Reinicia el servidor (CTRL+C y python main.py).

3._El índice FAISS se actualizará automáticamente si borras faiss_index.bin o cambias create_index(..., rebuild=True).
```
.gitignore recomendado
rag-env/
__pycache__/
*.pyc
.env
faiss_index.bin
```
**Contacto**
```
Proyecto desarrollado por Pedro Luzuriaga

Correo: pedriniandre@gmal.com
```
