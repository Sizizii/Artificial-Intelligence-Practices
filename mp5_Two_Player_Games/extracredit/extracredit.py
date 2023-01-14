import torch, random, math, json
import torch.nn as nn
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights
import pdb
DTYPE=torch.float32
DEVICE=torch.device("cpu")

in_channels = 15

###########################################################################################
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.flatten = torch.nn.Flatten()
        self.embedding1 = torch.nn.Linear(8*8*15,128)
        self.embedding2 = torch.nn.Linear(128,256)
        self.embedding3 = torch.nn.Linear(256, 512)
        self.embedding4 = torch.nn.Linear(512, 96)
        self.embedding5 = torch.nn.Linear(96,1)
        # self.embedding1.weight.data = initialize_weights()
        # self.embedding1.bias.data = torch.zeros(1)
        # self.embedding2 = torch.nn.Linear(1,8)
        # self.embedding7 = torch.nn.Linear(8,32)
        # self.embedding3 = torch.nn.Linear(32,64)
        # self.embedding4 = torch.nn.Linear(64,128)
        #self.embedding5 = torch.nn.Linear(128,64)
        #self.embedding6 = torch.nn.Linear(64,1)
        self.activation = torch.nn.ReLU()
        # pdb.set_trace()
    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.embedding1(x))
        x = self.activation(self.embedding2(x))
        x = self.activation(self.embedding3(x))
        x = self.activation(self.embedding4(x))
        x = self.embedding5(x)
        return x

def trainmodel():
    # Well, you might want to create a model a little better than this...
    #model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=20),torch.nn.ReLU(inplace=False),torch.nn.Linear(20,8), torch.nn.ReLU(inplace=False),torch.nn.Linear(in_features=8, out_features=1))
    model = nn.Sequential(nn.Flatten(), nn.Linear(8*8*15, 128),nn.ReLU(inplace = False),nn.Linear(128,256),nn.ReLU(inplace = False),nn.Linear(256,512),nn.ReLU(inplace = False),nn.Linear(512,96),nn.ReLU(inplace = False), nn.Linear(96,1))
    import pdb
    #pdb.set_trace()
    # ... and if you do, this initialization might not be relevant any more ...
   
    #model[1].weight.data = initialize_weights()
    #model[1].bias.data = torch.zeros(1)
    learning_rate = 3e-5
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))
    #loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_fn = torch.nn.L1Loss()
    loss_fn2 = torch.nn.MSELoss()

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    for epoch in range(10000):
        for x,y in trainloader:
            # pass # Replace this line with some code that actually does the training
            #pdb.set_trace()
            y_pred = model(x)
            loss = loss_fn(y_pred, y) + loss_fn2(y_pred, y)
            
            if epoch < 10:
                print(epoch, loss.item())
            elif epoch % 100 == 99:
                print(epoch, loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
