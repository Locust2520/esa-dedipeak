import torch
import torch.nn as nn
import torch.nn.functional as F


# Source: https://github.com/vincent-leguen/DILATE/blob/master/models/seq2seq.py


class EncoderRNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size):
        super(EncoderRNN, self).__init__()  
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)

    def forward(self, input, hidden): # input [batch_size, length T, dimensionality d]      
        output, hidden = self.gru(input, hidden)      
        return output, hidden
    
    def init_hidden(self, device):
        #[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_grulstm_layers, fc_units, output_size):
        super(DecoderRNN, self).__init__()      
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)         
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden) 
        output = F.relu( self.fc(output) )
        output = self.out(output)      
        return output, hidden


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder = EncoderRNN(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_grulstm_layers=1,
            batch_size=configs.batch_size
        )
        self.decoder = DecoderRNN(
            input_size=configs.dec_in,
            hidden_size=configs.d_model,
            num_grulstm_layers=1,
            fc_units=configs.d_model // 8,
            output_size=configs.c_out
        )
        self.target_length = configs.pred_len
        
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        x = batch_x
        input_length  = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(x.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:,ei:ei+1,:]  , encoder_hidden)
            
        decoder_input = x[:,-1,:].unsqueeze(1) # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]  ).to(x.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:,di:di+1,:] = decoder_output
        return outputs