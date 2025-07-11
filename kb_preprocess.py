import json
import chromadb

class PreprocessorKB:
    def __init__(self, persist_dir="./preprocess_kb_dir"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="preprocess_kb_dir")

    def load_knowledge(self, json_path):
        with open(json_path, 'r') as file:
            knowledge = json.load(file)

        documents = []
        metadatas = []
        ids = []

        for idx, entry in enumerate(knowledge):
            doc = json.dumps(entry)

            metadata = {
                "type": entry["type"].strip().lower()
            }

            doc_id = f"impute_{idx}"
            documents.append(doc)
            metadatas.append(metadata)
            ids.append(doc_id)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Imputation knowledge successfully stored in ChromaDB.")

    def search_knowledge(self, data_type):
        results = self.collection.query(
            query_texts=["imputation"],
            where={
                "type": data_type
            },
            n_results=1
        )

        if results['documents'] and results['documents'][0]:
            return results['documents'][0][0]
        else:
            return "No relevant imputation method found for the given type."


# json_file_path = "/content/preprocess_kb.json"

# kb = PreprocessorKB()
# kb.load_knowledge(json_file_path)

# print(kb.search_knowledge("numerical continuous"))
