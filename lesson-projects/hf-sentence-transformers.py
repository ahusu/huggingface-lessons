from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences1 = ['the cat sits outside ', 'a man is playing guitar', 'this movie is great']

embeddings = model.encode(sentences1, convert_to_tensor=True)

print(embeddings)