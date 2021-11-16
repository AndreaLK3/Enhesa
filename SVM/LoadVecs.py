import os
import Utils
import Filepaths
import re
import numpy as np

def test():
    vecs_fpath = os.path.join(Filepaths.vectors_folder, "Word2Vec_vectors.txt")
    with open(vecs_fpath, "r") as vecs_file:
        line = vecs_file.readline()
        line_ls = line.split()
        word_str = line_ls[0]

        word_1 = re.sub(r"^b'", '', word_str)
        word_2 = re.sub(r"'$", '', word_1)

        vector_str = line_ls[1:]
        vector_ls = []
        for num_str in vector_str:
            num = float(num_str[1:-1])
            vector_ls.append(num)
        vector_arr = np.array(vector_ls)
        print("*")
