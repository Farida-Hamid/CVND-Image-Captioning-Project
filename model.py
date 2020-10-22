import torch
import torch.nn as nn
import torchvision.models as models

############################################################################################################################
# A lot of this code is based on the article https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3 #
############################################################################################################################


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #nn.Embedding(num_embeddings: int, embedding_dim: int)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        #self.lstm = nn.LSTM(input_size, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
    
    def forward(self, features, captions):
        
        #Embed captions disregaring the <end> caption since it's used as a stop sign
        captions = captions[:,:-1]
        captions = self.embed(captions)
        
        # initialize the hidden and cell states to zeros
        batch_size = features.size(0)
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()
        
        x_out = torch.cat((features.unsqueeze(1), captions), dim=1)
        x_out, (h, c) = self.lstm(x_out)
        
        x_out = self.fc(x_out)
        
        return x_out#, h, c

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sent_out = []
        
        for i in range(max_len):
            x_out, states = self.lstm(inputs, states)    # calculate the LSTM output and the next state
            x_out = self.fc(x_out.squeeze(1))            # calculate the final output of the model
            
            best = x_out.max(1)[1]                       # choose the most probble next word
            sent_out.append(best.item())
            inputs = self.embed(best).unsqueeze(1)       
            
        return sent_out
            
        
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)