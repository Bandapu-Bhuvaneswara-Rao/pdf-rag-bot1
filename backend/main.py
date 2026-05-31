import os

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware

from rag import process_pdf
from rag import ask_question

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

@app.get("/")
def home():
    return {
        "message": "PDF RAG Bot API Running"
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as f:
        f.write(await file.read())

    process_pdf(file_path)

    return {
        "message": "PDF Uploaded Successfully"
    }


@app.post("/ask")
async def ask(data: dict):

    question = data["question"]

    answer = ask_question(question)

    return {
        "answer": answer
    }