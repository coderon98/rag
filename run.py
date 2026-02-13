import requests

OLLAMA_URL = "http://localhost:5000"
QDRANT_URL = "http://localhost:6333"

def embed(text):
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": "locusai/all-minilm-l6-v2",
            "prompt": text
        }
    )
    return r.json()["embedding"]

def search_qdrant(vector):
    r = requests.post(
        f"{QDRANT_URL}/collections/documents/points/search",
        json={
            "with_payload": True,
            "vector": vector,
            "limit": 3
        }
    )
    return r.json()["result"]

def generate(prompt):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "stream": False
        }
    )
    return r.json()["response"]

query = "c'est quoi une isochrone ?"

vector = embed(query)

results = search_qdrant(vector)
for r in results:
    print(r)
context = "\n".join([r["payload"]["text"] for r in results])

response = generate(f"""
Tu es un assistant technique précis.

Réponds uniquement avec les informations du contexte.
Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".
Contexte:
{context}

Question:
{query}
""")

print(response)
