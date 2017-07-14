import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

class RN(nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        
        self.img_enc = nn.Sequential(
                        nn.Conv2d(3, 24, 3, stride=2, padding=1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(24),
                        nn.Conv2d(24, 24, 3, stride=2, padding=1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(24),
                        nn.Conv2d(24, 24, 3, stride=2, padding=1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(24),
                        nn.Conv2d(24, 24, 3, stride=2, padding=1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(24))
        
        # Convolution image output is: B x 24 x 5 x 5
        # Before going to G, needs to be concatenated with 
        # question and coordinate
        
        self.G = nn.Sequential(
                        nn.Linear(2 * 26 + 11, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU())
        
        self.F = nn.Sequential(
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, 10),
                        nn.Softmax())
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.loss_func = nn.NLLLoss()
        
    
    def forward(self, img, qst):
        img_out = self.img_enc(img) # B x 24 x 5 x 5
        
        batch_size, n_chans, h, w = img_out.size()
        img_flat = img_out.view(batch_size, n_chans, h*w) # B x 24 x 25
        img_flat = img_flat.permute(0, 2, 1) # B x 25 x 24
        
        # Tag coordinates, B x 25 x 2
        coord_tensor = self.get_coord_tensor(batch_size, h, w)
        img_coord = torch.cat([img_flat, coord_tensor], dim = 2) # B x 25 x 26
        
        # Concat all possible object pairs
        x_i = torch.unsqueeze(img_coord, 1).repeat(1, h*w, 1, 1) # B x 25 x 25 x 26
        x_j = torch.unsqueeze(img_coord, 2).repeat(1, 1, h*w, 1) # B x 25 x 25 x 26
        x_full = torch.cat([x_i, x_j], dim=3)
        
        # Concat the same question to all object pairs
        # qst: B x 11
        # Goal: B x 25 x 25 x 11
        qst = torch.unsqueeze(qst, 1).repeat(1, (h*w)*(h*w), 1) # B x (25*25) x 11
        qst = qst.view(batch_size, h*w, h*w, 11) # B x 25 x 25 x 11
        
        
        x_full_qst = torch.cat([x_full, qst], 3) # B x 25 x 25 x (2 * 26 + 11)
        
        # Reshape into relations
        x_rel = x_full_qst.view(batch_size * (h*w)*(h*w), -1)
        
        # Object relation
        g_out = self.G(x_rel) # B * 25^2 x 256
        g_out = g_out.view(batch_size, (h*w)*(h*w), -1)
        g_out = torch.squeeze(torch.sum(g_out, dim=1))
        
        # Final
        final = self.F(g_out)
        return final
        
    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        
        loss = self.loss_func(output, label)
        loss.backward()
        
        self.optimizer.step()
        
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        
        return accuracy
        
    def get_coord_tensor(self, batch_size, h, w):
        np_coord_tensor = np.zeros((batch_size, h*w, 2))

        for i in range(w*h):
            # [(i/5-2)/2., (i%5-2)/2.]
            np_coord_tensor[:, i,:] = np.array([int(i / h), i % w])

        coord_tensor = torch.FloatTensor(batch_size, w*h, 2)
        coord_tensor = Variable(coord_tensor)    
        coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        
        return coord_tensor