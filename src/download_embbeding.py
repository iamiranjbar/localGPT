from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # or any other model of your choice

# Save the model locally
model.save("./local_sbert_model")
# model.save("../../tokenizer/local_sbert_model")
