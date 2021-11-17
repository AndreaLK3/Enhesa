import torch
from torch.nn.parameter import Parameter
import itertools
import Models.CorpusReader as CorpusReader
import Utils
import Models.LoadVecs as LV

# ------------- Define model -------------- #
class CNN(torch.nn.Module):

    def __init__(self, embeddings_matrix):
        super(CNN, self).__init__()

        self.E = Parameter(torch.FloatTensor(embeddings_matrix), requires_grad=True)
        dim_embs = self.E.shape[1]
        self.select_first_indices = Parameter(torch.tensor(list(range(1024))).to(torch.float32),
                                               requires_grad=False)  # can be used for select_index
        self.conv1d_k2 = torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=10,
                                             stride=1, padding=0, dilation=1, groups=1, bias=True,
                                             padding_mode='zeros')
        self.conv1d_k5 = torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=10,
                                         stride=1, padding=0, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        self.conv1d_k8 = torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=10,
                                         stride=1, padding=0, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        self.maxpool_n1 = torch.nn.MaxPool1d(kernel_size=200, stride=None, padding=0, dilation=1)

        self.fully_connected_layer = torch.nn.Linear(150, 100)


    def forward(self, indices_input_tensor, label):
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        # self.memory_hn.detach_()

        x_t = self.E[indices_input_tensor].t().unsqueeze(0)
        # e.g. if the article contains 68 words, x_t.shape=(batch_size=1, embeddings_dim=300, sequence_length=68)

        conv1 = self.conv1d_layer1(x_t)  # (1,100,59)
        intermediate_state1 = torch.nn.ReLU(conv1)
        maxpooled1 = self.maxpool_n1(conv1)  # (1,100,11)






def run_train():

    corpus_df = Utils.load_split(Utils.Split.TRAINING)
    word_embeddings = LV.get_word_vectors(False)
    model = CNN(word_embeddings)

    training_iterator = CorpusReader.next_featuresandlabel_article(corpus_df)

    for article_indices, article_label in training_iterator:
        x_indices_t = torch.tensor(article_indices)
        y_t = torch.tensor(article_label)
        model(x_indices_t, y_t)
