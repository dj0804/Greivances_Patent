from sentence_transformers import SentenceTransformer
def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save_pretrained('../Src/Models/sbert_local_model')

if __name__ == "__main__":
    main()