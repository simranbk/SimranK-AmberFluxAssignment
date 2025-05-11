from fastapi import FastAPI, File, UploadFile
from video_processor import extract_frames
from features import compute_color_histogram
from vector_store import init_collection, upload_vectors, search_similar
import os
import shutil
import glob
from qdrant_client.http.exceptions import UnexpectedResponse



app = FastAPI()
FRAME_DIR = "extracted_frames"

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_path = f"temp_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("Video saved:", video_path)

        extract_frames(video_path, FRAME_DIR)
        print("Frames extracted to:", FRAME_DIR)

        vectors = []
        image_paths = []
        for img_path in sorted(glob.glob(f"{FRAME_DIR}/*.jpg")):
            vec = compute_color_histogram(img_path)
            vectors.append(vec)
            image_paths.append(img_path)

        print("Feature vectors computed.")
        # try:
        #     client.get_collection(COLLECTION_NAME)
        # except UnexpectedResponse:
        #     init_collection(vector_size=len(vectors[0]))
        init_collection(vector_size=len(vectors[0]))
        print("Qdrant collection initialized.")
        
        upload_vectors(vectors, image_paths)
        print("Vectors uploaded.")

        os.remove(video_path)
        return {"status": "uploaded and processed", "frames": len(vectors)}

    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

@app.post("/search/")
async def search_similar_frames(file: UploadFile = File(...)):
    with open("query.jpg", "wb") as f:
        shutil.copyfileobj(file.file, f)
    vec = compute_color_histogram("query.jpg")
    results = search_similar(vec)
    print("Search vector length:", len(vec))
    print("Vector:", vec)
    # print("Total vectors in Qdrant:", client.count(COLLECTION_NAME).count)

    print(f"Found {len(results)} matches:", results)
    for res in results:
        print("Score:", res.score, "Image:", res.payload["path"])

    return [
        {
            "score": r.score,
            "image": r.payload["path"]
        } for r in results
    ]
