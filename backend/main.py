import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.inference import VQAInference
from backend.nlg import generate_explanation

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_last.pth")

app = FastAPI(title="PathVQA-TR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vqa = None


@app.on_event("startup")
def load_model():
    global vqa
    if not os.path.exists(MODEL_PATH):
        print(f"UYARI: Model dosyası bulunamadı: {MODEL_PATH}. Önce eğitimi çalıştırın.")
        return
    vqa = VQAInference(MODEL_PATH)
    print("Model yüklendi.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": vqa is not None}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    if vqa is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Görüntü boş.")

    vqa_result = vqa.predict(image_bytes, question)

    explanation = generate_explanation(
        user_question=question,
        vqa_answer=vqa_result["answer"],
        confidence=vqa_result["confidence"],
        translated_question=vqa_result["translated_question"],
    )

    return {
        "answer": vqa_result["answer"],
        "confidence": vqa_result["confidence"],
        "translated_question": vqa_result["translated_question"],
        "explanation": explanation,
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")