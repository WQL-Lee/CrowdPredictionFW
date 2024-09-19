import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
from base import BaseModel
# from model.layers import *
# from layers import *
import numpy as np
import argparse as arg
from torch import Tensor


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        '''

        :param x: [B, T, N ,C]
        :return:
        '''

        if self.c_in > self.c_out:
            x = x.permute(0, 3, 1, 2)
            x_align = self.align_conv(x)
            x_align = x_align.permute(0, 2, 3, 1)
        elif self.c_in < self.c_out:
            batch_size, T, n_vertex, c_in = x.shape
            x_align = torch.cat([x, torch.zeros([batch_size, T, n_vertex, self.c_out - self.c_in]).to(x)], dim=-1)
        else:
            x_align = x

        return x_align
    
class OutputLayer(nn.Module):
    def __init__(self, T, c_in, c_out, n_vertex, act_func='GLU'):
        '''

        :param T:
        :param c_in:
        :param c_out: output_dim : 1
        :param n_vertex:
        :param act_func:
        '''
        super(OutputLayer, self).__init__()
        # map T to 1
        self.tmpconv_in = TemporalConvLayer(T, c_in, c_in, n_vertex, act_func)
        self.tmpconv_out = TemporalConvLayer(1, c_in, c_in, n_vertex, 'sigmoid')
        self.layernorm = nn.LayerNorm([n_vertex, c_in])
        # map c_in to c_out, keep T
        # [B, T, N , C_in] -> [B, T, N, C_out]
        self.fully_con = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = self.tmpconv_in(x)
        x = self.layernorm(x)
        x = self.tmpconv_out(x).permute(0, 3, 1, 2)
        x = self.fully_con(x).permute(0, 2, 3, 1)

        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func='relu'):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.act_func = act_func
        if act_func == 'GLU':
            # self.w_t = nn.Parameter(torch.FloatTensor(Kt,1,c_in,2*c_out),requires_grad=True)
            # self.b_t= nn.Parameter(torch.FloatTensor(2*c_out),requires_grad=True)
            self.tempconv = nn.Conv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), padding=(0, 0),
                                      dilation=(1, 1))
        else:
            # self.w_t = nn.Parameter(torch.FloatTensor(Kt, 1, c_in, c_out), requires_grad=True)
            # self.b_t = nn.Parameter(torch.FloatTensor(c_out), requires_grad=True)
            self.tempconv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), padding=(0, 0),
                                      dilation=(1, 1))

    def forward(self, x):
        _, T, n, _ = x.shape

        x_in = self.align(x)[:, self.Kt - 1:T, :, :]  ##resnet

        # BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        x_conv = self.tempconv(x).permute(0, 2, 3, 1)
        # x_conv=F.conv2d(x,self.w_t,self.b_t,stride=[1,1])

        if self.act_func == 'GLU':
            return (x_conv[:, :, :, 0:self.c_out] + x_in) * torch.sigmoid(x_conv[:, :, :, self.c_out:])
        elif self.act_func == 'linear':
            return x_conv
        elif self.act_func == 'sigmoid':
            return torch.sigmoid(x_conv)

        elif self.act_func == 'relu':
            return torch.relu(x_conv + x_in)

        else:
            raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')

class ConvBlock(nn.Module):
    def __init__(self,  Kt, channels, n_vertex, dropout=1, act_func='GLU', graph_act_func='relu'):
        super(ConvBlock, self).__init__()

        self.Kt = Kt
        # self.dropout=dropout
        self.act_func = act_func
        self.c_si, self.c_t, self.c_oo = channels
        self.n_vertex = n_vertex

        self.tmp1conv = TemporalConvLayer(Kt, self.c_si, self.c_t, n_vertex, act_func=self.act_func)
        self.tmp2conv = TemporalConvLayer(self.Kt, self.c_t, self.c_oo, n_vertex, self.act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, self.c_oo])

        if graph_act_func == 'relu':
            self.graph_act_func = nn.ReLU()
        elif graph_act_func == 'sigmoid':
            self.graph_act_func = nn.Sigmoid()
        elif graph_act_func == 'tanh':
            self.graph_act_func = nn.Tanh()
        elif graph_act_func == 'leaky_relu':
            self.graph_act_func = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.tmp1conv(x)
        x = self.graph_act_func(x)
        x = self.tmp2conv(x)
        x = self.tc2_ln(x)
        return self.dropout(x)




class Crowd_CNN_GRU(BaseModel):
    def __init__(self, Kt, his, n_pred, info_foresee, n_vertex, global_ch, gru_hidden, gru_layers, out_size,
                 temp_act_func, block_act_func, flight_cin, flight_act_func, out_act_func, dropout):
        super(Crowd_CNN_GRU, self).__init__()
        self.n_vertex = n_vertex
        self.stconvblock = nn.ModuleList()
        self.his = his
        self.n_pred = n_pred
        self.info_foresee = info_foresee
        self.drop_out = dropout
        self.out_size = out_size
        self.n_pred = n_pred
        self.name = "Crowd_CNN_GRU"

        Ko = his
        for i, ch in enumerate(global_ch):
            # TODO:if you change the num of temporal block,you may change the '2' here.
            Ko -= 2 * (Kt - 1)
            self.stconvblock.append(
                ConvBlock(Kt, ch, n_vertex, dropout, temp_act_func, block_act_func)
            )
        if Ko <= 1:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')
        else:
            self.output_layer = OutputLayer(Ko, global_ch[-1][-1], out_size,
                                            n_vertex, out_act_func)
        # self.flight_fusion = Flight_Extrator_GRU(flight_cin, n_vertex, info_foresee, flight_act_func)

        # self.fusion_linear_1 = nn.Linear(2*n_vertex,n_vertex)
        # self.fusion_linear_2 = nn.Linear(n_vertex, n_vertex)

        self.gru_layers = gru_layers
        self.gru_hidden = gru_hidden
        self.rnn_cell = nn.ModuleList()
        self.rnn_cell.append(nn.GRUCell(out_size, gru_hidden))

        for _ in range(gru_layers - 1):
            self.rnn_cell.append(nn.GRUCell(gru_hidden, gru_hidden))

        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.gru_hidden if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

    def forward(self, x_g, t_g, flight):
        """
        Teacher Forcing...
        :param x_l: [batch_size, T, n_vertex_local, channel]
        :param x_g: [batch_size, T, n_vertex_global, channel]
        :param flight : [batch_size, n_vertex, N_in]
        :param t_g: [batch_size, n_pred, n_vertex_global, channel]
        :return: [batch_size , n_pred, n_vertex, out_channel]
        """
        B, n_pred, N, _ = x_g.shape
        # B, n_pred, _, N_cin, flight_C = flight.shape
        for i, block in enumerate(self.stconvblock):
            x_g = block(x_g)
        x_g = self.output_layer(x_g)
        #
        # flight_x = self.flight_fusion(flight.view(-1, self.info_foresee, N_cin))
        # # [B, T, N]
        # flight_x = flight_x.view(B, n_pred, N)

        y_all = []

        h_list = [x_g.new_zeros(B*self.n_vertex,self.gru_hidden)]
        for l in range(1, self.gru_layers):
            h_list.append(x_g.new_zeros(B*self.n_vertex,self.gru_hidden))

        #
        # go_frame = t_g.new_zeros(B, 1, N)
        # t_g = torch.cat([go_frame, t_g.squeeze(-1)], dim=1)

        for t in range(self.n_pred):
            rnn_input = x_g.squeeze(1).reshape(-1, self.out_size)
            # rnn_input = x_g.squeeze() + flight_x[:, t, :]
            h_list[0] = self.rnn_cell[0](
                rnn_input, h_list[0])
            for l in range(1, self.gru_layers):
                h_list[l] = nn.functional.dropout(self.rnn_cell[l](
                    h_list[l - 1], h_list[l]), self.drop_out)
            rnn_output = h_list[-1]
            predicted_y_t = self.mlp(rnn_output)
            y_all.append(predicted_y_t)

        y_all = torch.stack(y_all, dim=1)

        y_all = y_all.view(-1, self.n_vertex, self.n_pred, 1)
        y_all = y_all.permute(0, 2, 1, 3)

        return y_all

    def decode(self, x_g, flight):
        """
        :param x_l: [batch_size, T, n_vertex_local, channel]
        :param x_g: [batch_size, T, n_vertex_global, channel]
        :param flight : [batch_size, n_vertex, N_in]
        :param t_g: [batch_size, n_pred, n_vertex_global, channel]
        :return: [batch_size , n_pred, n_vertex, out_channel]
        """
        B, n_pred, N, _ = x_g.shape
        # B, n_pred, _, N_cin, flight_C = flight.shape
        for i, block in enumerate(self.stconvblock):
            x_g = block(x_g)
        x_g = self.output_layer(x_g)

        # flight_x = self.flight_fusion(flight.view(-1, self.info_foresee, N_cin))
        # # [B, T, N]
        # flight_x = flight_x.view(B, n_pred, N)

        y_all = []

        h_list = [x_g.new_zeros(B*self.n_vertex,self.gru_hidden)]
        for l in range(1, self.gru_layers):
            h_list.append(x_g.new_zeros(B*self.n_vertex,self.gru_hidden))


        # go_frame = x_g.new_zeros(B, 1, N)
        # y_all.append(go_frame)

        for t in range(self.n_pred):
            rnn_input = x_g.squeeze(1).reshape(-1, self.out_size)
            # rnn_input = x_g.squeeze() + flight_x[:, t, :]
            h_list[0] = self.rnn_cell[0](
                rnn_input, h_list[0])
            for l in range(1, self.gru_layers):
                h_list[l] = nn.functional.dropout(self.rnn_cell[l](
                    h_list[l - 1], h_list[l]), self.drop_out)
            rnn_output = h_list[-1]
            predicted_y_t = self.mlp(rnn_output)
            y_all.append(predicted_y_t)

        y_all = torch.stack(y_all, dim=1)

        y_all = y_all.view(-1, self.n_vertex, self.n_pred, 1)
        y_all = y_all.permute(0, 2, 1, 3)

        return y_all
    
    
if __name__=="__main__":
    flight = torch.rand((50, 12, 12, 20))
    x_t = torch.rand((50, 12, 26, 1))
    t_g = torch.rand((50, 12, 26, 1))
    model = Crowd_CNN_GRU(3, 12, 12, 12, 26, [[1, 16, 16]], 80, 1, 1, 'GLU', 'relu', 20,
                                              'relu', 'GLU', 0)
    _ = model(x_t, t_g, flight)
    print(f"{_.shape}")
