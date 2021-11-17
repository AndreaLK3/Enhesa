import torch
from torch.nn import Parameter, functional as tfunc


class ConvNet(torch.nn.Module):

    def __init__(self, embeddings_matrix, num_classes):
        super(ConvNet, self).__init__()

        self.E = Parameter(torch.FloatTensor(embeddings_matrix), requires_grad=True)
        dim_embs = self.E.shape[1]
        self.select_first_indices = Parameter(torch.tensor(list(range(1024))).to(torch.float32),
                                               requires_grad=False)  # can be used for select_index
        self.conv1d_k3 = torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
        self.conv1d_k5 = torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=5)
        self.conv1d_k8 = torch.nn.Conv1d(in_channels=300, out_channels=100, kernel_size=8)
       # The global maxpooling operation is handled via torch.nn.functional

        self.linear2classes = torch.nn.Linear(300, num_classes)


    def forward(self, indices_input_tensor, label):
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        # self.memory_hn.detach_()

        x_t = self.E[indices_input_tensor].t().unsqueeze(0)
        # e.g. if the article contains 68 words, x_t.shape=(batch_size=1, embeddings_dim=300, sequence_length=68)

        conv1_k3 = tfunc.tanh(self.conv1d_k3(x_t))  # (1,100,66)
        conv1_k5 = tfunc.tanh(self.conv1d_k5(x_t))  # (1,100,64)
        conv1_k8 = tfunc.tanh(self.conv1d_k8(x_t))  # (1,100,61)

        features_k3 = tfunc.max_pool1d(conv1_k3, kernel_size=conv1_k3.shape[2])  # (1,100,1)
        features_k5 = tfunc.max_pool1d(conv1_k5, kernel_size=conv1_k5.shape[2])  # (1,100,1)
        features_k8 = tfunc.max_pool1d(conv1_k8, kernel_size=conv1_k8.shape[2])  # (1,100,1)

        doc_rep = torch.cat([features_k3,features_k5,features_k8], dim=1).squeeze(2)  # (1,300)

        logits_classes = self.linear2classes(doc_rep)
        predictions_classes = tfunc.log_softmax(logits_classes, dim=1)

        return predictions_classes