import fitz
import pytesseract
from PIL import Image
import io
import pathlib
import uuid
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

BASE_DIR = pathlib.Path("/home/rdesouza/Bureau/rag")

DOCS_DIR = BASE_DIR / "docs"
PROCESSED_DIR = BASE_DIR / "processed"
IMAGE_DIR = BASE_DIR / "images"

DOCS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "documents"

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to Qdrant...")
client = QdrantClient(url="http://localhost:6333")

# create collection if not exists
try:
    client.get_collection(COLLECTION_NAME)
except:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

def extract_pdf(pdf_path):

    print(f"Extracting {pdf_path}")

    doc = fitz.open(pdf_path)
    full_text = []

    for page_index, page in enumerate(doc):

        text = page.get_text()

        if text.strip():
            full_text.append(text)

        images = page.get_images(full=True)

        for img_index, img in enumerate(images):

            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))

            image_filename = IMAGE_DIR / f"{pdf_path.stem}_p{page_index}_{img_index}.png"
            image.save(image_filename)

            ocr_text = pytesseract.image_to_string(
                image,
                lang="eng+fra"
            )

            if ocr_text.strip():
                full_text.append(ocr_text)

    return "\n".join(full_text)


def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[start:start+chunk_size]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def index_document(pdf_path):

    print(f"Indexing {pdf_path}")

    text = extract_pdf(pdf_path)

    if not text.strip():
        return

    chunks = chunk_text(text)

    embeddings = embedding_model.encode(chunks)

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": pdf_path.name
                }
            )
        )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    pdf_path.rename(PROCESSED_DIR / pdf_path.name)

    print("Indexed successfully")


def extract_txt(txt_path: pathlib.Path) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def index_txt_document(txt_path: pathlib.Path):

    print(f"Indexing {txt_path}")

    # 1. extraire texte
    text = extract_txt(txt_path)

    if not text.strip():
        print("Empty file, skipping")
        return

    # 2. découper en chunks
    chunks = chunk_text(text)

    if not chunks:
        print("No chunks generated")
        return

    # 3. générer embeddings
    embeddings = embedding_model.encode(chunks)

    # 4. créer points Qdrant
    points = []

    for chunk, embedding in zip(chunks, embeddings):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": txt_path.name,
                    "type": "txt"
                }
            )
        )

    # 5. insérer dans Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    # 6. déplacer fichier vers processed
    txt_path.rename(PROCESSED_DIR / txt_path.name)

    print("Indexed successfully")


class Handler(FileSystemEventHandler):

    def on_created(self, event):

        if event.src_path.endswith(".pdf"):

            time.sleep(2)
            index_document(pathlib.Path(event.src_path))


def scan_existing():
    for txt in DOCS_DIR.glob("*.txt"):
        index_txt_document(txt)


    for pdf in DOCS_DIR.glob("*.pdf"):
        index_document(pdf)


if __name__ == "__main__":

    scan_existing()

    observer = Observer()

    observer.schedule(
        Handler(),
        str(DOCS_DIR),
        recursive=False
    )

    observer.start()

    print("Watching docs folder...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
