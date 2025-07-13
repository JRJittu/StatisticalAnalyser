import json
import chromadb

class StatisticalKnowledgeBase:
    def __init__(self, persist_dir="stat_kb_dir"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="stat_kb_dir")

    def load_knowledge(self, json_path):
        with open(json_path, 'r') as file:
            knowledge = json.load(file)

        documents = []
        metadatas = []
        ids = []

        for idx, entry in enumerate(knowledge['statistical_tests']):
            doc = json.dumps(entry)
            metadata = {
                "no_of_variable": entry['no_of_variable'],
                "var_type": entry['var_type']
            }
            doc_id = f"stat_test_{idx}"
            documents.append(doc)
            metadatas.append(metadata)
            ids.append(doc_id)

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print("Knowledge successfully stored in ChromaDB.")

    def search_knowledge(self, no_of_variable, var_type):
        results = self.collection.query(
            query_texts=["statistical test"],  # placeholder
            where={
                "$and": [
                    {"no_of_variable": no_of_variable},
                    {"var_type": var_type}
                ]
            },
            n_results=1
        )

        if results['documents'] and results['documents'][0]:
            return results['documents'][0][0]  # return first matched document
        else:
            return "No relevant statistical test found."

