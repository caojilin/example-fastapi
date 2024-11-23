from collections import defaultdict
from fastapi import FastAPI, Query
import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from rapidfuzz import process, fuzz
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

app = FastAPI()
pc = Pinecone(
    api_key="pcsk_368Tpa_ULGYwn74G43mBVq1hEUaaReNWtM38AFjwhU3cpJjKvpXqaV4GxJDGWaY12tagMR")

index = pc.Index("the-office")


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

    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "passage", "truncate": "END"}
    )

    # Convert the query into a numerical vector that Pinecone can search with
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )

    # Search the index for the three most similar vectors
    results = index.query(
        namespace="example-namespace",
        vector=query_embedding[0].values,
        top_k=limit,
        include_values=False,
        include_metadata=True
    )

    rows = [int(result['id'])-1 for result in results['matches']]
    scores = [result['score'] for result in results['matches']]

    sentence_match = []
    for i in range(len(rows)):
        sentence_match.append(
            (rows[i], scores[i]*100, rows[i]))

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
