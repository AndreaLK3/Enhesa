import os


dataset_folder = "10kGNAD"
training_file = os.path.join(dataset_folder, "train.csv")
test_file = os.path.join(dataset_folder, "test.csv")

images_folder = os.path.join("ExploreDataset", "Figures")
vectors_folder = "Word2Vec_vectors"
vectors_fname = "German_Word2Vec_vectors.txt"

models_folder = "Models"
vocabulary_fname = "vocabularyCounter.pickle"
vocabulary_fpath = os.path.join(models_folder, vocabulary_fname)
random_wordEmb_fname = "WordEmbeddings_randomNormal.npy"
random_wordEmb_fpath = os.path.join(models_folder, random_wordEmb_fname)
pretrained_wordEmb_fname = "WordEmbeddings_pretrained.npy"
pretrained_wordEmb_fpath = os.path.join(models_folder, random_wordEmb_fname)


tests_folder = "Tests"
mytests_file = os.path.join(tests_folder, "MyTest.csv")
mytests_vectors_fname = "mini_Word2Vec_vectors.txt"
mytests_vectors_wordslist_fname = "Word2Vec_vectors_wordslist.txt"