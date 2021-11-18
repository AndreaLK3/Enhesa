import torch
from torch.nn import Parameter, functional as tfunc


class ConvNet(torch.nn.Module):

    def __init__(self, embeddings_matrix, num_classes):
        super(ConvNet, self).__init__()

        self.E = Parameter(torch.FloatTensor(embeddings_matrix), requires_grad=True)
        dim_embs = self.E.shape[1]
        dim_conv_out = 100
        self.conv1d_k3 = torch.nn.Conv1d(in_channels=dim_embs, out_channels=dim_conv_out, kernel_size=3, padding=2)
        self.conv1d_k6 = torch.nn.Conv1d(in_channels=dim_embs, out_channels=dim_conv_out, kernel_size=6, padding=5)

       # The global maxpooling operation is handled via torch.nn.functional

        self.linear2classes = torch.nn.Linear(dim_conv_out*2, num_classes)


    def forward(self, indices_input_tensor):
        # CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        x_t = self.E[indices_input_tensor].t().unsqueeze(0)
        # e.g. if the article contains 68 words, x_t.shape=(batch_size=1, embeddings_dim=300, sequence_length=68)

        conv1_k3 = torch.tanh(self.conv1d_k3(x_t))  # (1,100,66)
        conv1_k6 = torch.tanh(self.conv1d_k6(x_t))  # (1,100,63)

        features_k3 = tfunc.max_pool1d(conv1_k3, kernel_size=conv1_k3.shape[2])  # (1,100,1)
        features_k6 = tfunc.max_pool1d(conv1_k6, kernel_size=conv1_k6.shape[2])  # (1,100,1)

        doc_rep = torch.cat([features_k3,features_k6], dim=1).squeeze(2)  # (1,300)

        logits_classes = self.linear2classes(doc_rep)
        predictions_classes = tfunc.log_softmax(logits_classes, dim=1)

        return predictions_classes