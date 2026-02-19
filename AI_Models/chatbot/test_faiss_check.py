from vector_db_embedding import *

def check_faiss(db_path, name):
    print("\n========================")
    print(f"DB 검사: {name}")
    print("========================")

    db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("총 벡터 수:", db.index.ntotal)

    docs = db.similarity_search("로그인", k=3)

    for i, doc in enumerate(docs):
        print(f"\n--- 검색 결과 {i+1} ---")
        print(doc.page_content[:300])
        print("metadata:", doc.metadata)


if __name__ == "__main__":
    check_faiss(IMAGE_FAISS_DIR, "IMAGE")
    check_faiss(API_FAISS_DIR, "API")
    check_faiss(TEXT_FAISS_DIR, "TEXT")