from src.python.utils import ProjectUtils


x = ProjectUtils.load_preprocessed_pubmed_data()

# y = set()
# for i in range(500):
#     y.add(x[i]['id'][:-4])
#
# print(y)
# print(len(y))

# for i, text in enumerate(x[:500]):
#     print(i, ' | ', text)


model = ProjectUtils.load_word2vec_embeddings()
word_vectors = model['computer']  # Get vector for 'computer'
similar_words = model.most_similar('computer')  # Find similar words to 'computer'
print(word_vectors)
print(similar_words)

