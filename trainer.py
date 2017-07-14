import torch
from torch.autograd import Variable
import numpy as np

class Trainer():
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        
        self.input_img = torch.FloatTensor(batch_size, 3, 75, 75)
        self.input_qst = torch.FloatTensor(batch_size, 11)
        self.label = torch.LongTensor(batch_size)

        self.input_img = Variable(self.input_img)
        self.input_qst = Variable(self.input_qst)
        self.label = Variable(self.label)
        
    def check_data(self, data):
        assert len(data) == 3
        print(len(data[0]), len(data[1]), len(data[2]))
        print(data[0][0].shape)
        print(data[1][0].shape)
        print(data[2][0])
    
    def tensor_data(self, data, i):
        img = torch.from_numpy(np.asarray(data[0][self.batch_size * i : self.batch_size * (i+1)]))
        qst = torch.from_numpy(np.asarray(data[1][self.batch_size * i : self.batch_size * (i+1)]))
        ans = torch.from_numpy(np.asarray(data[2][self.batch_size * i : self.batch_size * (i+1)]))

        self.input_img.data.resize_(img.size()).copy_(img)
        self.input_qst.data.resize_(qst.size()).copy_(qst)
        self.label.data.resize_(ans.size()).copy_(ans)    
    
    def train(self, rank, model, data, epochs):
        torch.manual_seed(rank)
        
        accuracy_history = []
        
        for epoch in range(epochs):
            best_accuracy = -float('inf')
            
            for i in range(int(len(data[0]) / self.batch_size)):
                self.tensor_data(data, i)
                accuracy = model.train_(self.input_img, self.input_qst, self.label)
                accuracy_history.append(accuracy)
                
                best_accuracy = max(accuracy, best_accuracy)

                if i % 500 == 0:
                    print('Epoch {}/{} Batch {}/{} Accuracy {:.0f}%'. 
                          format(epoch, epochs, i, len(data[0]) / self.batch_size, best_accuracy))
                
            print("Epoch {} / {} Best accuracy is: {:.0f}".format(epoch, epochs, best_accuracy))
        
        accuracy_history = np.array(accuracy_history)
        np.save('loss_history-{}.npy'.format(rank), accuracy_history)