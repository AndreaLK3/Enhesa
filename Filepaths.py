import os


dataset_folder = "10kGNAD"
training_file = os.path.join(dataset_folder, "train.csv")
test_file = os.path.join(dataset_folder, "test.csv")

images_folder = os.path.join("ExploreDataset", "Figures")
vectors_folder = "Word2Vec_vectors"
vectors_fname = "German_Word2Vec_vectors.txt"
vectors_vocab_fname = "German_Word2Vec_vocabulary.txt"

models_folder = "Models"
vocabulary_fname = "vocabulary.txt"
vocabulary_file = os.path.join(models_folder, vocabulary_fname)

tests_folder = "Tests"
mytests_file = os.path.join(tests_folder, "MyTest.csv")
mytests_vectors_fname = "mini_Word2Vec_vectors.txt"
mytests_vectors_wordslist_fname = "Word2Vec_vectors_wordslist.txt"