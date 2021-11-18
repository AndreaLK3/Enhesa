import os


dataset_folder = "10kGNAD"
train_file = os.path.join(dataset_folder, "train.csv")
test_file = os.path.join(dataset_folder, "test.csv")

images_folder = os.path.join("DatasetGraphics", "Figures")
vectors_folder = "Word2Vec_vectors"
vectors_fname = "German_Word2Vec_vectors.txt"

models_folder = "Model"
saved_models_subfolder = "SavedModels"
resources_subfolder = "Resources"
measures_to_display_subfolder = "Measures to display"

vocabulary_fname = "vocabularyList.pickle"
vocabulary_fpath = os.path.join(models_folder, resources_subfolder, vocabulary_fname)

random_wordEmb_fname = "WordEmbeddings_randomNormal.npy"
random_wordEmb_fpath = os.path.join(models_folder, resources_subfolder, random_wordEmb_fname)
pretrained_wordEmb_fname = "WordEmbeddings_pretrained.npy"
# pretrained_wordEmb_fpath = os.path.join(models_folder, resources_subfolder, pretrained_wordEmb_fname)
# pre-trained Word2Vec embeddings are not in use in the latest version, due to space constraints

training_set_file = os.path.join(models_folder, resources_subfolder, "training.csv")
validation_set_file = os.path.join(models_folder, resources_subfolder, "validation.csv")

tests_folder = "MyTests"
mytests_file = os.path.join(tests_folder, "MyTest.csv")
mytests_vectors_fname = "mini_Word2Vec_vectors.txt"
mytests_vectors_wordslist_fname = "Word2Vec_vectors_wordslist.txt"