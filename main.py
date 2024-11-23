from collections import defaultdict
from fastapi import FastAPI, Query
import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# Get the folder where this script is located
current_folder = Path(__file__).resolve().parent
csv_file_path = current_folder / "schrute.csv"

df = pd.read_csv(csv_file_path, header=0)
df = df.drop("Unnamed: 0", axis=1)
print(df.columns)

df['lower'] = df['text'].str.lower()
df['lower'] = df['lower'].str.strip()
# Replace empty strings with NaN and then drop those rows
df.replace(pd.NA, "", inplace=True)
# df.replace("", pd.NA, inplace=True)
# df.dropna(inplace=True)
# df = df.reset_index(drop=True)

# Load the array from the file
with open(current_folder / "all_embeddings.pkl", "rb") as f:
    all_embeddings = pickle.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

origins = [
    "http://localhost",
    "http://localhost:8889",
    "https://caojilin-playground.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI"}


@app.get("/api/py/rapidfuzz")
def rapidfuzz(query: str, limit: int = 5):
    query = query.lower().strip()
    limit = int(limit)
    exact_match = [(text, 100, i)
                   for i, text in enumerate(df["lower"]) if query in text][:limit]
    fuzz_match = process.extract(
        query, choices=df["lower"], scorer=fuzz.partial_ratio, score_cutoff=0.8, limit=limit)

    query_embeddings = model.encode(query)
    similarity_scores = util.pytorch_cos_sim(
        all_embeddings, query_embeddings).view(-1)
    top_indices = np.argsort(similarity_scores).tolist()[::-1]

    sentence_match = []
    for row in top_indices[:limit]:
        sentence_match.append(
            (row, round(similarity_scores[row].tolist(), 2)*100, row))

    return_body = {}
    counter = 0
    return_body, counter = convert_to_json(
        return_body, counter, exact_match, 'exact match')
    return_body, counter = convert_to_json(
        return_body, counter, fuzz_match, 'partial ratio')
    return_body, counter = convert_to_json(
        return_body, counter, sentence_match, 'sentence embedding')
    return return_body


def convert_to_json(return_body, counter, arr, name):

    for _, result in enumerate(arr):
        _, score, row = result
        _, season, episode, episode_name, director, writer, character, text, _, _ = df.loc[row].tolist(
        )
        return_body[counter] = {
            "season": int(season),  # Ensure it's a Python int
            "episode": int(episode),
            "episode_name": str(episode_name),
            "director": str(director),
            "writer": str(writer),
            "character": str(character),
            "text": str(text),
            "score": float(score),
            "method": name
        }
        counter += 1
    return return_body, counter

