# Use matplotlib to display some visualizations on the dataset:
# - Number of articles per class. How much is the dataset imbalanced? (bar plot)
# - Average number of words in an article, per class? (bar plot)
# - How much overlap there is between the vocabulary of any two classes? (color matrix)
# - How much of the vocabulary of a class is unique, i.e. non-overlapping? (bar plot)

import DatasetGraphics.ExploreDataset as ED

ED.all_visualizations()