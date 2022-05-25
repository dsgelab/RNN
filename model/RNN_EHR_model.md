

```python
"""
RNN model for predictions

"""
from __future__ import print_function, division
from io import open
import string
import re
import random
import os
import argparse
import time
import math
import torch
  
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    

#sys.path.insert() only for jupyter notebook imports
import sys
sys.path.insert(0, '/data/project_avabalas/RNN')


#silly ones
from termcolor import colored
from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

#DATASET DATALOADER classes
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
plt.ion()
try:
    import cPickle as pickle
except:
    import pickle    
from tabulate import tabulate


from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

#added by me
import pandas as pd
import linecache
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
```

    Using cuda:2 device



```python
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

## DATASET and DATALOADER classes


```python
def splitdata(root_dir, file, nm, case_drop, test_ratio = 0, valid_ratio = 0):


    filepath = root_dir + file
    data = pd.read_csv(filepath, usecols=['FINREGISTRYID','LABEL'])


    #data_zeroes = data[data['LABEL']==0]
    #data_ones = data[data['LABEL']==1]
    ones_indices = data.index[data['LABEL']==1].values
    np.random.shuffle(ones_indices)
    data2 = data.drop(ones_indices[:int(len(ones_indices) * case_drop)]).copy()   #to create a new DataFrame, if you want to modify the original one, put inplace=True

    if nm != 5000000:
        _, X_test, _, y_test = train_test_split(data2, data2['LABEL'], stratify=data2['LABEL'], test_size=nm)
    else:
        X_test = data2
    cc_ratio = X_test[X_test['LABEL']==1].shape[0]/X_test[X_test['LABEL']==0].shape[0]
    print("case control ratio =  ",cc_ratio)
    train, val_test, y_train, y_test = train_test_split(X_test, X_test['LABEL'], stratify=X_test['LABEL'], test_size=test_ratio+valid_ratio)
    test, valid, y_test, y_valid = train_test_split(val_test, val_test['LABEL'], stratify=val_test['LABEL'], test_size=valid_ratio/(test_ratio+valid_ratio))

    train = train.index.to_list()
    valid = valid.index.to_list()
    test = test.index.to_list()

    return train, test, valid, cc_ratio

class EHRdataFromCsv(Dataset):
    def __init__(self, root_dir, list_IDs, file = None, transform=None, model='RNN'):

        self.file = None
        if file != None:
            self.file = file
            self.filepath = root_dir + file
            self.list_IDs = list_IDs
        else:
            print('No file specified')
    
    def __getitem__(self, idx, seeDescription = False):

        if self.file != None: 
            row = self.list_IDs[idx]
        else:
            print('No file specified')
            
        sample = linecache.getline(self.filepath, row+2)[:-1].split(',')
        
        #n_seq=[]
        #for v in range(len(a[2].split(" "))):
        #    nv =[]
        #    nv.append([int(a[2].split(" ")[v])])
        #    codes=a[3].split(" ")
        #    nv.append(list(map(int, codes[v].split(";"))))
        #    n_seq.append(nv)
        #sample = [a[0],int(a[1]),n_seq,list(map(float,a[6:]))]
          
        return sample

    def __len__(self):
        ''' 
        just the length of data
        '''
        if self.file != None:
            return len(self.list_IDs)
        else: 
            print('No file specified')

    
def preprocess(batch,pack_pad,cc_ratio): 
    # Check cuda availability

    flt_typ=torch.FloatTensor
    lnt_typ=torch.LongTensor

    mb=[]
    mtd=[]
    lbt=[]
    seq_l=[]
    dem =[]
    idd_t =[]
    fset=[]
    for samp in range(len(batch)):
        n_seq=[]
        ages = batch[samp][2].split(' ')
        codes = batch[samp][3].split(' ')
        for v in range(len(codes)):
            nv=[]
            nv.append([int(ages[v])])
            visit_codes = list(map(int, codes[v].split(";")))
            nv.append(visit_codes)                      
            n_seq.append(nv)
        n_pt= [batch[samp][0],int(batch[samp][1]),n_seq,list(map(float,batch[samp][5].split())),int(batch[samp][4])]
        fset.append(n_pt)
    batch = fset
    #assert False, "breakpoint"
    
    bsize=len(batch) ## number of patients in minibatch

    batch.sort(key=lambda pt:pt[4],reverse=True)  

    #llv=0
    #for x in batch:
    #    lv= len(max(x[2], key=lambda xmb: len(xmb[1]))[1])
    #    if llv < lv:
    #        llv=lv     # max number of codes per visit in minibatch    
    llv=100     # max number of codes per visit in minibatch
    lp = max(batch, key=lambda xmb: xmb[4])[4] ## maximum number of visits per patients in minibatch
    
    for pt in batch:
        sk,label,ehr_seq_l,demo,lpx = pt # lpx no of visits in pt record

        seq_l.append(lpx) 
        lbt.append(flt_typ([[float(label)]]))
        dem.append(torch.from_numpy(np.asarray(demo,dtype=float)).type(flt_typ))
        idd_t.append(sk)
        ehr_seq_tl=[]
        time_dim=[]

        for ehr_seq in ehr_seq_l:
            pd=(0, (llv -len(ehr_seq[1])))
            result = F.pad(torch.from_numpy(np.asarray(ehr_seq[1],dtype=int)).type(lnt_typ),pd,"constant", 0)
            ehr_seq_tl.append(result)
            time_dim.append(torch.from_numpy(np.asarray(ehr_seq[0],dtype=int)).type(flt_typ))

        ehr_seq_t= torch.stack(ehr_seq_tl,0) 
        #assert False, "breakpoint"
        lpp= lp-lpx ## diffence between max seq in minibatch and cnt of patient visits 
        if pack_pad:
            zp= nn.ZeroPad2d((0,0,0,lpp)) ## (0,0,0,lpp) when use the pack padded seq and (0,0,lpp,0) otherwise. 
        else: 
            zp= nn.ZeroPad2d((0,0,lpp,0))
        ehr_seq_t= zp(ehr_seq_t) ## zero pad the visits med codes
        mb.append(ehr_seq_t)
        time_dim_v= torch.stack(time_dim,0)
        time_dim_pv= zp(time_dim_v) ## zero pad the visits time diff codes
        #time_dim_pv=time_dim_pv.to(device)
        mtd.append(time_dim_pv)
    lbt_t= torch.stack(lbt,0)
    mb_t= torch.stack(mb,0)
    dem_t= torch.stack(dem,0)

    weight = lbt_t.clone()
    weight[weight==0]=torch.tensor(cc_ratio)
    
    #lbt_t=lbt_t.to(device)
    #mb_t=mb_t.to(device)
    #dem_t=dem_t.to(device)
    #weight=weight.to(device)
    


    return mb_t, lbt_t,seq_l, mtd, dem_t, idd_t, weight
            
    
    
         
#customized parts for EHRdataloader
def my_collate(batch):
    mb_t, lbt_t,seq_l, mtd, dem_t,idd_t,weight  =preprocess(batch,pack_pad,cc_ratio)
    return [mb_t, lbt_t,seq_l, mtd, dem_t,idd_t,weight]
            

def iter_batch2(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(iterator.__next__())
    random.shuffle(results)  
    return results

class EHRdataloader(DataLoader):
    def __init__(self, dataset, cc_ratio, batch_size=128, num_workers=0,  shuffle=True, sampler=None, batch_sampler=None, collate_fn=my_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, packPadMode = False):
        DataLoader.__init__(self, dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, sampler=None, batch_sampler=None, collate_fn=my_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
        self.collate_fn = collate_fn
        global pack_pad
        pack_pad = packPadMode
```

## Embedding class


```python
class EHREmbeddings(nn.Module):
    #initialization and then the forward and things
    #DRNN has no bi, QRNN no bi, TLSTM has no bi, but DRNN has other cell-types 
    #cel_type are different for each model variation 
    def __init__(self, input_size, embed_dim ,hidden_size,use_demo, n_layers=1,dropout_r=0.1,cell_type='LSTM', bii=False, time=False , preTrainEmb='', packPadMode = True):
        super(EHREmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.use_demo = use_demo
        self.n_layers = n_layers
        self.dropout_r = dropout_r
        self.cell_type = cell_type
        self.time=time
        self.preTrainEmb=preTrainEmb
        self.packPadMode = packPadMode
        if bii: 
            self.bi=2 
        else: 
            self.bi=1
            
        if len(input_size)==1:
            self.multi_emb=False
            if len(self.preTrainEmb)>0:
                emb_t= torch.FloatTensor(np.asmatrix(self.preTrainEmb))
                self.embed= nn.Embedding.from_pretrained(emb_t)#,freeze=False) 
                self.in_size= embed_dim ### need to be updated to be automatically capyured from the input
            else:
                input_size=input_size[0]
                self.embed= nn.Embedding(input_size, self.embed_dim,padding_idx=0)#,scale_grad_by_freq=True)
                self.in_size= embed_dim
        else:
            if len(input_size)!=3: 
                raise ValueError('the input list is 1 length')
            else: 
                self.multi_emb=True
                self.diag=self.med=self.oth=1

        #self.emb = self.embed.weight  LR commented Jul 10 19
        if self.time: self.in_size= self.in_size+1 
               
        if self.cell_type == "GRU":
            self.cell = nn.GRU
        elif self.cell_type == "RNN":
            self.cell = nn.RNN
        elif self.cell_type == "LSTM":
            self.cell = nn.LSTM
        elif self.cell_type == "QRNN":
            from torchqrnn import QRNN
            self.cell = QRNN
        elif self.cell_type == "TLSTM":
            from tplstm import TPLSTM
            self.cell = TPLSTM 
        else:
            raise NotImplementedError
       
        if self.cell_type == "QRNN": 
            self.bi=1 ### QRNN not support Bidirectional, DRNN should not be BiDirectional either.
            self.rnn_c = self.cell(self.in_size, self.hidden_size, num_layers= self.n_layers, dropout= self.dropout_r)
        elif self.cell_type == "TLSTM":
            self.bi=1 
            self.rnn_c = self.cell(self.in_size, hidden_size)
        else:
            self.rnn_c = self.cell(self.in_size, self.hidden_size, num_layers=self.n_layers, dropout= self.dropout_r, bidirectional=bii, batch_first=True)
        if use_demo: 
            self.out = nn.Linear(self.hidden_size*self.bi+92,1)
        else:
            self.out = nn.Linear(self.hidden_size*self.bi,1)
        self.sigmoid = nn.Sigmoid()

      
                            
    #let's define this class method
    def EmbedPatients_MB(self,mb_t, mtd): #let's define this
        self.bsize=len(mb_t) ## no of pts in minibatch
        embedded = self.embed(mb_t)  ## Embedding for codes
        #assert False, "breakpoint"
        embedded = torch.sum(embedded, dim=2) 
        if self.time:
            mtd_t= Variable(torch.stack(mtd,0))
            mtd_t = ((mtd_t-39.4166)/21.6504)*4
            #mtd_t = mtd_t*100000
            mtd_t.to(device)
            out_emb= torch.cat((embedded,mtd_t),dim=2)
            
        else:
            out_emb= embedded
        out_emb.to(device)
        #assert False, "breakpoint"
        return out_emb
    
```










## MODEL classes


```python
## only the class EHR_RNN is tetsted atm


class EHR_RNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, use_demo, n_layers=1,dropout_r=0.1,cell_type='GRU',bii=False ,time=False, preTrainEmb='',packPadMode = True):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, use_demo, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, bii=bii, time=time , preTrainEmb=preTrainEmb, packPadMode=packPadMode)



    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def init_hidden(self):
        
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        h_0.to(device)
        if self.cell_type == "LSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
        return result
    
    def forward(self, input, x_lens, mtd, demo, use_demo):
        
        x_in  = self.EmbedPatient_MB(input, mtd) 
        ### uncomment the below lines if you like to initiate hidden to random instead of Zero which is the default
        #h_0= self.init_hidden()
        #h_0.to(device)

        if self.packPadMode: 
            x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)#, enforce_sorted=False)   
            output, hidden = self.rnn_c(x_inp)#,h_0) 
        else:
            output, hidden = self.rnn_c(x_in)#,h_0) 
        #assert False, "breakpoint"
        if self.cell_type == "LSTM":
            hidden=hidden[0]
        #assert False, "breakpoint" 
        if self.bi==2:
            if use_demo:
                fc = self.out( torch.cat(  (torch.cat((hidden[-2],hidden[-1]),1),demo)   ,1) )
            else:
                fc = self.out(torch.cat((hidden[-2],hidden[-1]),1))

            output = self.sigmoid(fc)
        else:
            if use_demo:
                fc = self.out(torch.cat(   (hidden[-1],demo)   ,1) )
            else:
                fc = self.out(hidden[-1])
            
            
            output = self.sigmoid(fc)
        
        return output #.squeeze()

#Model 2: DRNN, DGRU, DLSTM
class EHR_DRNN(EHREmbeddings): 
    def __init__(self,input_size,embed_dim, hidden_size, n_layers, dropout_r=0.1,cell_type='GRU', bii=False, time=False, preTrainEmb='', packPadMode = False):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type,bii=False, time=time , preTrainEmb=preTrainEmb, packPadMode=False)


        self.dilations = [2 ** i for i in range(n_layers)]
        self.layers = nn.ModuleList([])
        if self.bi == 2:
            print('DRNN only supports 1-direction, implementing 1-direction instead')
        self.bi =1  #Enforcing 1-directional
        self.packPadMode = False #Enforcing no packpadded indicator 
        
        for i in range(n_layers):
            if i == 0:
                c = self.cell(self.in_size, self.hidden_size, dropout=self.dropout_r)
            else:
                c = self.cell(self.hidden_size, self.hidden_size, dropout=self.dropout_r)
            self.layers.append(c)
        self.cells = nn.Sequential(*self.layers)
        #self.out = nn.Linear(hidden_size,1)
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
        
## Uncomment if using the mutiple embeddings version ---not included in this release    
#     def EmbedPatient_SMB(self, input,mtd):
#         return EHREmbeddings.EmbedPatients_SMB(self, input,mtd)     
    
    def forward(self, input,  x_lens, mtd, hidden=None):
        x  = self.EmbedPatient_MB(input,mtd) 

        x=x.permute(1,0,2)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                x,_ = self.drnn_layer(cell, x, dilation)
            else:
                x,hidden[i] = self.drnn_layer(cell, x, dilation, hidden[i]) 
            
        outputs=x[-dilation:]
        x=self.sigmoid(self.out(torch.sum(outputs,0))) #change from F to self.sigmoid, should be the same
        return x.squeeze()

        
######Dilated RNN related methods
    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = inputs.size(0)
        batch_size = inputs.size(1)
        hidden_size = cell.hidden_size
        #print('hidden size',hidden_size) --verified correct

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)
        return outputs, hidden
       
    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):

        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):

        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):

        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):

        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))

            zeros_ = zeros_.to(device)

            inputs = torch.cat((inputs, Variable(zeros_)))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps


    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs
    
    def init_hidden(self, batch_size, hidden_size):
        c = Variable(torch.zeros(batch_size, hidden_size))

        c = c.to(device)
        if self.cell_type == "LSTM":
            m = Variable(torch.zeros(batch_size, hidden_size))   
            m = m.to(device)
            return (c, m)
        else:
            return c        


# Model 3: QRNN
class EHR_QRNN(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers =1 ,dropout_r=0.1, cell_type='QRNN', bii=False, time=False, preTrainEmb='', packPadMode = False):

       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, bii=bii, time=time , preTrainEmb=preTrainEmb, packPadMode=packPadMode)

        #super(EHR_QRNN, self).__init__()
        #basically, we dont allow cell_type and bii choices
        #let's enfroce these:
        if (self.cell_type !='QRNN' or self.bi !=1):
            print('QRNN only supports 1-direction & QRNN cell_type implementation. Implementing corrected parameters instead')
        self.cell_type = 'QRNN'
        self.bi = 1 #enforcing 1 directional
        self.packPadMode = False #enforcing correct packpaddedmode
        
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def forward(self, input, x_lens, mtd):    
        x_in  = self.EmbedPatient_MB(input,mtd) 
        x_in = x_in.permute(1,0,2) ## QRNN not support batch first
        output, hidden = self.rnn_c(x_in)#,h_0) 
        output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze()



# Model 4: T-LSTM
class EHR_TLSTM(EHREmbeddings):
    def __init__(self,input_size,embed_dim, hidden_size, n_layers =1 ,dropout_r=0.1, cell_type='TLSTM', bii=False, time=True, preTrainEmb=''):

        EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers, dropout_r, cell_type, time , preTrainEmb)
       	EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size, n_layers=n_layers, dropout_r=dropout_r, cell_type=cell_type, bii=False, time=True , preTrainEmb=preTrainEmb, packPadMode=False)
        
        if self.cell_type !='TLSTM' or self.bi != 1:
            print("TLSTM only supports Time aware LSTM cell type and 1 direction. Implementing corrected parameters instead")
        self.cell_type = 'TLSTM'
        self.bi = 1 #enforcing 1 directional
        self.packPadMode = False

        
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    def EmbedPatient_SMB(self, input,mtd):
        return EHREmbeddings.EmbedPatients_SMB(self, input, mtd)   

  
    def init_hidden(self):
        h_0 = Variable(torch.rand(self.n_layers*self.bi,self.bsize, self.hidden_size))
        h_0= h_0.to(device)
        if self.cell_type == "LSTM"or self.cell_type == "TLSTM":
            result = (h_0,h_0)
        else: 
            result = h_0
        return result
   
    
    def forward(self, input, x_lens, mtd):
        x_in  = self.EmbedPatient_MB(input,mtd) 
        x_in = x_in.permute(1,0,2) 
        #x_inp = nn.utils.rnn.pack_padded_sequence(x_in,x_lens,batch_first=True)### not well tested
        h_0 = self.init_hidden()
        output, hidden,_ = self.rnn_c(x_in,h_0) 
        if self.cell_type == "LSTM" or self.cell_type == "TLSTM":
            hidden=hidden[0]
        if self.bi==2:
            output = self.sigmoid(self.out(torch.cat((hidden[-2],hidden[-1]),1)))
        else:
            output = self.sigmoid(self.out(hidden[-1]))
        return output.squeeze()

# Model 5: Logistic regression (with embeddings):
class EHR_LR_emb(EHREmbeddings):
    def __init__(self, input_size,embed_dim,use_demo, time=False, cell_type= 'LR',preTrainEmb=''):
        
         EHREmbeddings.__init__(self,input_size, embed_dim ,hidden_size = embed_dim,use_demo=use_demo)
         
    #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
## Uncomment if using the mutiple embeddings version ---not included in this release    
#     def EmbedPatient_SMB(self, input,mtd):
#         return EHREmbeddings.EmbedPatients_SMB(self, input,mtd)     
    def forward(self, input, x_lens, mtd, demo, use_demo):
#         if self.multi_emb:
#             x_in  = self.EmbedPatient_SMB(input,mtd)
#         else: 
        x_in  = self.EmbedPatient_MB(input,mtd)
        if use_demo:
            fc = self.out(torch.cat(   (torch.sum(x_in,1),demo)   ,1) )
        else:
            fc = self.out(torch.sum(x_in,1))
        output = self.sigmoid(fc)
        return output #.squeeze()


# Model 6:Retain Model
class RETAIN(EHREmbeddings):
    def __init__(self, input_size, embed_dim, hidden_size, n_layers):
        
        EHREmbeddings.__init__(self,input_size = input_size, embed_dim=embed_dim ,hidden_size=hidden_size)
        self.embed_dim = embed_dim
        self.RNN1 = nn.RNN(embed_dim,hidden_size,1,batch_first=True,bidirectional=True)
        self.RNN2 = nn.RNN(embed_dim,hidden_size,1,batch_first=True,bidirectional=True)
        self.wa = nn.Linear(hidden_size*2,1,bias=False)
        self.Wb = nn.Linear(hidden_size*2,hidden_size,bias=False)
        self.W_out = nn.Linear(hidden_size,n_layers,bias=False)
        self.sigmoid = nn.Sigmoid()
        #embedding function goes here 
    def EmbedPatient_MB(self, input, mtd):
        return EHREmbeddings.EmbedPatients_MB(self, input, mtd)
    
    
    def forward(self, input, x_lens, mtd):
        # get embedding using self.emb
        b = len(input)
        x_in  = self.EmbedPatient_MB(input,mtd) 
            
        h_0 = Variable(torch.rand(2,self.bsize, self.hidden_size))
        x_in = x_in.to(device)
        h_0 = h_0.to(device)
      
        # get alpha coefficients
        outputs1 = self.RNN1(x_in,h_0) # [b x seq x 128*2]
        b,seq,_ = outputs1[0].shape
        E = self.wa(outputs1[0].contiguous().view(-1, self.hidden_size*2)) # [b*seq x 1]     
        alpha = F.softmax(E.view(b,seq),1) # [b x seq]
        self.alpha = alpha
         
        # get beta coefficients
        outputs2 = self.RNN2(x_in,h_0) # [b x seq x 128]
        b,seq,_ = outputs2[0].shape
        outputs2 = self.Wb(outputs2[0].contiguous().view(-1,self.hidden_size*2)) # [b*seq x hid]
        self.Beta = torch.tanh(outputs2).view(b, seq, self.embed_dim) # [b x seq x 128]
        result = self.compute(x_in, self.Beta, alpha)
        return result.squeeze()

    # multiply to inputs
    def compute(self, embedded, Beta, alpha):
        b,seq,_ = embedded.size()
        outputs = (embedded*Beta)*alpha.unsqueeze(2).expand(b,seq,self.embed_dim)
        outputs = outputs.sum(1) # [b x hidden]
        return self.sigmoid(self.W_out(outputs)) # [b x num_classes]
    
    
    # interpret
    def interpret(self,u,v,i,o):
        # u: user number, v: visit number, i: input element number, o: output sickness
        a = self.alpha[u][v] # [1]
        B = self.Beta[u][v] # [h] embed dim
        W_emb = self.emb[i] # [h] embed)dim
        W = self.W_out.weight.squeeze() # [h]
        out = a*torch.dot(W,(B*W_emb))
        return out
```

## UTILITIES functions


```python
###### minor functions, plots and prints

#loss plot
def showPlot(points):
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.show()

#auc_plot 
def auc_plot(y_real, y_hat):
    fpr, tpr, _ = roc_curve(y_real,  y_hat)
    auc = roc_auc_score(y_real, y_hat)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show();

        
#time Elapsed
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)    


#print to file function
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


###### major model training utilities
def trainsample(sample, label_tensor, seq_l, mtd, demo, model, optimizer,criterion, weighted_loss, cc_ratio, use_demo, weight): 
    model.train() ## LR added Jul 10, that is the right implementation
    model.zero_grad()
    #demo = torch.squeeze(label_tensor).unsqueeze(1) #XXX
    #demo = (demo-demo.mean())/demo.std()
    #assert False, "breakpoint"
    output = model(sample,seq_l, mtd, demo, use_demo)
    del sample,seq_l, mtd, demo
    #assert False, "breakpoint"
    if weighted_loss:
        #weight = label_tensor.clone()
        #weight[weight==0]=torch.tensor(cc_ratio).to(device)
        criterion = nn.BCELoss(weight=weight)
    loss = criterion(output.unsqueeze(dim=1), label_tensor)
    if weighted_loss:
        del weight
    #assert False, "breakpoint"
    loss.backward()   
    optimizer.step()
    # print(loss.item())
    #assert False, "breakpoint"
    return output, loss.item()


#train with loaders

def trainbatches(mbs_list, model, optimizer, weighted_loss, cc_ratio, use_demo, shuffle = True):#,we dont need this print print_every = 10, plot_every = 5): 
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    #if shuffle: 
        # you can also shuffle batches using iter_batch2 method in EHRDataloader
        #  loader = iter_batch2(mbs_list, len(mbs_list))
    #assert False, "breakpoint"
        #random.shuffle(mbs_list)        
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mt, demo, _, weight = batch
        sample=sample.to(device)
        label_tensor=label_tensor.to(device)
        demo=demo.to(device)
        mtd =[]
        for nnz in mt:
            mtd.append(nnz.to(device))
        weight=weight.to(device)
        #assert False, "breakpoint"
        output, loss = trainsample(sample, label_tensor, seq_l, mtd, demo, model, optimizer, criterion = nn.BCELoss(), weighted_loss=weighted_loss, cc_ratio=cc_ratio, use_demo=use_demo, weight=weight)
        del sample, label_tensor, seq_l, mtd, demo, weight
        current_loss += float(loss)
        n_iter +=1
        
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0
    #assert False, "breakpoint"
    return current_loss, all_losses 



def calculate_auc(model, mbs_list, which_model, shuffle, use_demo): # batch_size= 128 not needed
    model.eval() ## LR added Jul 10, that is the right implementation
    y_real =[]
    y_hat= []
    idds = []
    #idds = np.array([]).reshape(-1,1)
    #if shuffle: 
    #    random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mt, demo, iddss, _ = batch
        sample=sample.to(device)
        label_tensor=label_tensor.to(device)
        demo=demo.to(device)
        mtd =[]
        for nnz in mt:
            mtd.append(nnz.to(device))
        #demo=demo.to(device)
        #demo = torch.squeeze(label_tensor).unsqueeze(1) #XXX
        #demo = (demo-demo.mean())/demo.std()
        output = model(sample, seq_l, mtd, demo, use_demo)
        del sample, seq_l, mtd, demo
        y_hat.extend(output.cpu().data.view(-1).numpy())  
        y_real.extend(label_tensor.cpu().data.view(-1).numpy())
        idds.extend(iddss) 
    auc = roc_auc_score(y_real, y_hat)
    return auc, y_real, y_hat, idds 

    
#define the final epochs running, use the different names

def epochs_run(epochs, train, valid, test, model, optimizer, shuffle, which_model, patience, output_dir,weighted_loss, cc_ratio, use_demo, model_prefix = 'dhf.train', model_customed= ''): 
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    #header = 'BestValidAUC|TestAUC|atEpoch'
    #logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    #print2file(header, logFile)
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = trainbatches(mbs_list = train, model= model, optimizer = optimizer, weighted_loss=weighted_loss, cc_ratio=cc_ratio, use_demo=use_demo)

        train_time = timeSince(start)
        #epoch_loss.append(train_loss)
        avg_loss = np.mean(train_loss)
        valid_start = time.time()
        train_auc, _, _, _ = calculate_auc(model = model, mbs_list = train, which_model = which_model, shuffle = shuffle,use_demo=use_demo)
        valid_auc, _, _, _ = calculate_auc(model = model, mbs_list = valid, which_model = which_model, shuffle = shuffle,use_demo=use_demo)
        valid_time = timeSince(valid_start)
        print(colored('\n Epoch (%s): Train_auc (%s), Valid_auc (%s) ,Training Average_loss (%s), Train_time (%s), Eval_time (%s)'%(ep, train_auc, valid_auc , avg_loss,train_time, valid_time), 'green'))
        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            best_model= model 
            if test:      
                testeval_start = time.time()
                bestTestAuc, y_real, y_hat, idds = calculate_auc(model = best_model, mbs_list = test, which_model = which_model, shuffle = shuffle,use_demo=use_demo)
                auc_plot(y_real, y_hat)
                y_hat2 = np.round(np.clip(y_hat, 0, 1)).tolist()
                print(confusion_matrix(y_real,y_hat2))
                #assert False, "breakpoint"
                print(colored('\n Test_AUC (%s) , Test_eval_time (%s) '%(bestTestAuc, timeSince(testeval_start)), 'yellow')) 
                #print(best_model,model) ## to verify that the hyperparameters already impacting the model definition
                #print(optimizer)
        #if ep - bestValidEpoch > patience:
        #      break
    #if test:      
    #   bestTestAuc, _, _ = calculate_auc(model = best_model, mbs_list = test, which_model = which_model, shuffle = shuffle) ## LR code reorder Jul 10
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #save model & parameters
    torch.save(best_model, output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    torch.save(best_model.state_dict(), output_dir + model_prefix + model_customed + 'EHRmodel.st')
    '''
    #later you can do to load previously trained model:
    best_model= torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    best_model.load_state_dict(torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.st'))
    best_model.eval()
    '''
    #Record in the log file , modify below as you like, this is just as example
    header = 'BestValidAUC|TestAUC|atEpoch'
    logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    print2file(header, logFile)
    pFile = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch)
    print2file(pFile, logFile) 
    if test:
        print(colored('BestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch),'green'))
    else: 
        print(colored('BestValidAuc %f at epoch %d ' % (bestValidAuc,  bestValidEpoch),'green'))
        print('No Test Accuracy')
    print(colored('Details see ../models/%sEHRmodel.log' %(model_prefix + model_customed),'green'))
    return y_real, y_hat, idds

    
```


```python
#args, slightly modified from main.py file to be more compatible with jupyter notebook 
#all args provide default values, so you can run the whole notebook without changing/providing any args
#args ordered by dataloader, model, and training sections
def options():
    parser = argparse.ArgumentParser(description='Predictive Analytics on EHR with Pytorch')
    
    #EHRdataloader 
    parser.add_argument('-root_dir', type = str, default = '/data/projects/project_avabalas/RNN/preprocessing_new/' , 
                        help='the path to the folders with pickled file(s)')
    parser.add_argument('-file', type = str, default = 'grouped_DF.csv' , 
                        help='the name of pickled files')
    parser.add_argument('-test_ratio', type = float, default = 0.2, 
                        help='test data size [default: 0.2]')
    parser.add_argument('-valid_ratio', type = float, default = 0.1, 
                        help='validation data size [default: 0.1]')
    
    #EHRmodel
    parser.add_argument('-which_model', type = str, default = 'DRNN', 
                        help='choose from {"RNN","DRNN","QRNN","LR"}') 
    parser.add_argument('-cell_type', type = str, default = 'GRU', 
                        help='For RNN based models, choose from {"RNN", "GRU", "LSTM"}')
    parser.add_argument('-input_size', type = list, default =[15817], 
                        help='''input dimension(s), decide which embedding types to use. 
                        If len of 1, then  1 embedding; 
                        len of 3, embedding medical, diagnosis and others separately (3 embeddings) 
                        [default:[15817]]''') ###multiple embeddings not effective in this release
    parser.add_argument('-embed_dim', type=int, default=128, 
                        help='number of embedding dimension [default: 128]')
    parser.add_argument('-hidden_size', type=int, default=128, 
                        help='size of hidden layers [default: 128]')
    parser.add_argument('-dropout_r', type=float, default=0.1, 
                        help='the probability for dropout[default: 0.1]')
    parser.add_argument('-n_layers', type=int, default=3, 
                        help='''number of Layers, 
                        for Dilated RNNs, dilations will increase exponentialy with mumber of layers [default: 1]''')
    parser.add_argument('-bii', type=bool, default=False, 
                        help='indicator of whether Bi-directin is activated. [default: False]')
    parser.add_argument('-time', type=bool, default=False, 
                        help='indicator of whether time is incorporated into embedding. [default: False]')
    parser.add_argument('-preTrainEmb', type= str, default='', 
                        help='path to pretrained embeddings file. [default:'']')
    parser.add_argument("-output_dir",type=str, default= '/data/projects/project_avabalas/RNN/models', 
                        help="The output directory where the best model will be saved and logs written [default: we will create'../models/'] ")
    
    # training 
    parser.add_argument('-lr', type=float, default=10**-4, 
                        help='learning rate [default: 0.0001]')
    parser.add_argument('-L2', type=float, default=10**-4, 
                        help='L2 regularization [default: 0.0001]')
    parser.add_argument('-epochs', type=int, default= 100, 
                        help='number of epochs for training [default: 100]')
    parser.add_argument('-patience', type=int, default= 20, 
                        help='number of stagnant epochs to wait before terminating training [default: 20]')
    parser.add_argument('-batch_size', type=int, default=128, 
                        help='batch size for training, validation or test [default: 128]')
    parser.add_argument('-optimizer', type=str, default='adam', 
                        choices=  ['adam','adadelta','adagrad', 'adamax', 'asgd','rmsprop', 'rprop', 'sgd'], 
                        help='Select which optimizer to train [default: adam]. Upper/lower case does not matter')
    parser.add_argument('-weighted_loss', type=bool, default=True, 
                        help='if True weighs loss function so that both classes are represented equaly based on case/control ratio') 
    parser.add_argument('-use_demo', type=bool, default=True, 
                        help='if True demographic features are cncatinated to fc layer')
    parser.add_argument('-num_workers', type=int, default=0, 
                        help='number of DataLoader workers [default: 0]')
    #parser.add_argument('-cuda', type= bool, default=True, help='whether GPU is available [default:True]')
    args = parser.parse_args([])
    return args 
```

## StepX: You can modify parameters here to suit your own need


```python
args = options()
##Update the args here if you dont want to use the default ones
##start an example
args.batch_size = 200
args.which_model = 'RNN'
args.cell_type = 'GRU'
args.embed_dim = 128
args.hidden_size = 128
args.dropout_r = 0.5
args.n_layers = 2
args.input_size=[8600] # total number of different longitudinal codes
args.patience=20
args.epochs=1
args.time=True
args.weighted_loss=True
args.use_demo=True
args.lr = 0.001
args.L2 = 0.001
args.num_workers = 5
args.file = 'grouped_DF_all_codes_100codes.all.csv' #'grouped_DF_100codes.csv' / grouped_DF_100codes.all2.csv
##end
print(args)
linecache.clearcache()
#torch.cuda.empty_cache()
#linecache.clearcache()
```

    Namespace(L2=0.001, batch_size=200, bii=False, cell_type='GRU', dropout_r=0.5, embed_dim=128, epochs=1, file='grouped_DF_all_codes_100codes.all.csv', hidden_size=128, input_size=[8600], lr=0.001, n_layers=2, num_workers=5, optimizer='adam', output_dir='/data/projects/project_avabalas/RNN/models', patience=20, preTrainEmb='', root_dir='/data/projects/project_avabalas/RNN/preprocessing_new/', test_ratio=0.2, time=True, use_demo=True, valid_ratio=0.1, weighted_loss=True, which_model='RNN')



```python
results = np.zeros([1,7])
case_drop = 0

for nm in [i for i in [1000000] for _ in range(1)]: # Run code using subsample of size 1000000 for X loops (to use a full sample enter 5000000)
    #depending on different models, model parameters might have different choices.
    #e.g. if you set bi = True for DRNN or QRNN, it will throw you warnings and implement correct bi =False instead
    if args.which_model == 'RNN': 
        ehr_model = EHR_RNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  use_demo = args.use_demo,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r,
                                  cell_type=args.cell_type,
                                  bii= args.bii,
                                  time= args.time,
                                  preTrainEmb= args.preTrainEmb)

        pack_pad = True
    elif args.which_model == 'DRNN': 
        ehr_model = EHR_DRNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r, #default =0 
                                  cell_type=args.cell_type, #default ='DRNN'
                                  bii= False,
                                  time = args.time, 
                                  preTrainEmb= args.preTrainEmb,
                                  use_demo = args.use_demo)     
        pack_pad = False
    elif args.which_model == 'QRNN': 
        ehr_model = EHR_QRNN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r, #default =0.1
                                  cell_type= 'QRNN', #doesn't support normal cell types
                                  bii= False, #QRNN doesn't support bi
                                  time = args.time,
                                  preTrainEmb= args.preTrainEmb,
                                  use_demo = args.use_demo) 
        pack_pad = False
    elif args.which_model == 'TLSTM': 
        ehr_model = EHR_TLSTM(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  dropout_r=args.dropout_r, #default =0.1
                                  cell_type= 'TLSTM', #doesn't support normal cell types
                                  bii= False, 
                                  time = args.time, 
                                  preTrainEmb= args.preTrainEmb, 
                                  use_demo = args.use_demo)     
        pack_pad = False
    elif args.which_model == 'RETAIN': 
        ehr_model = RETAIN(input_size= args.input_size, 
                                  embed_dim=args.embed_dim, 
                                  hidden_size= args.hidden_size,
                                  n_layers= args.n_layers,
                                  use_demo = args.use_demo) 
        pack_pad = False
    else: 
        ehr_model = EHR_LR_emb(input_size = args.input_size,
                                     embed_dim = args.embed_dim,
                                     use_demo = args.use_demo,
                                     preTrainEmb= args.preTrainEmb)
        pack_pad = False


    #make sure cuda is working
    ehr_model = ehr_model.to(device)
    #model optimizers to choose from. Upper/lower case dont matter
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(ehr_model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.L2)
    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), 
                                   lr=args.lr, 
                                   weight_decay=args.L2)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), 
                                  lr=args.lr, 
                                  weight_decay=args.L2) 
    elif args.optimizer.lower() == 'adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.L2)
    elif args.optimizer.lower() == 'asgd':
        optimizer = optim.ASGD(ehr_model.parameters(), 
                               lr=args.lr, 
                               weight_decay=args.L2)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(ehr_model.parameters(), 
                                  lr=args.lr, 
                                  weight_decay=args.L2)
    elif args.optimizer.lower() == 'rprop':
        optimizer = optim.Rprop(ehr_model.parameters(), 
                                lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(ehr_model.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.L2)
    else:
        raise NotImplementedError
    
    print(ehr_model)
    
####### LOAD DATA ###############
    
    train, test, valid, cc_ratio = splitdata(root_dir = args.root_dir,                                            
                                             file = args.file,
                                             nm = nm,
                                             case_drop = case_drop,
                                             test_ratio = args.test_ratio,
                                             valid_ratio = args.valid_ratio)                                             
    #a = data.__getitem__(0, seeDescription = False)
    print(colored("\nSample data lengths for train, test and validation:", 'green'))
    print(len(train), len(test), len(valid))
    
    
    train_set = EHRdataFromCsv(root_dir = args.root_dir, list_IDs=train, file = args.file) 
    test_set = EHRdataFromCsv(root_dir = args.root_dir, list_IDs=test, file = args.file) 
    valid_set = EHRdataFromCsv(root_dir = args.root_dir, list_IDs=valid, file = args.file) 
    
    
    ##### separate loader for train, test, validation
    #if you have different files, you need to load them separately into EHRdataFromPickles()
    #and then use EHRdataloader() on each
    #dataloader's default will sort data based on length of visits and then split into batches with default batch_size/of your choice
    #new in this release is the creation of minibatches lists once before the epochs run, then will shuffle within the epochs
    train_mbs = EHRdataloader(train_set,cc_ratio, batch_size = args.batch_size, num_workers = args.num_workers, packPadMode = pack_pad, drop_last=True)
    valid_mbs = EHRdataloader(valid_set,cc_ratio, batch_size = args.batch_size, num_workers = args.num_workers, packPadMode = pack_pad, drop_last=True)
    test_mbs = EHRdataloader(test_set,cc_ratio, batch_size = args.batch_size, num_workers = args.num_workers, packPadMode = pack_pad, drop_last=True) 
    
####### MODEL TRAINING ###############

    #Notes: default: sort data based on visit length within batch only
    #default: （batch）shuffle = true
    #allows for keyboard interrupt
    #saving best model in the directory specified in args.output_dir
    #cc_ratio = 0.0244
    #print("case control ratio =  ",cc_ratio)

    try:
           y_real, y_hat, idds = epochs_run(args.epochs,
                                            train = train_mbs, 
                                            valid = valid_mbs, 
                                            test = test_mbs, 
                                            model = ehr_model, 
                                            optimizer = optimizer,
                                            shuffle = True, 
                                            which_model = args.which_model, 
                                            patience = args.patience,
                                            output_dir = args.output_dir,
                                            weighted_loss = args.weighted_loss,
                                            cc_ratio=cc_ratio,
                                            use_demo = args.use_demo)
    #we can keyboard interupt now 
    except KeyboardInterrupt:
        print(colored('-' * 89, 'green'))
        print(colored('Exiting from training early','green'))
    
    # record perormance
    auc = roc_auc_score(y_real, y_hat)*10000
    y_hat2 = np.round(np.clip(y_hat, 0, 1)).tolist()
    cf = confusion_matrix(y_real,y_hat2)
    TN = cf[0][0]
    TP = cf[1][1]
    FP = cf[0][1]
    FN = cf[1][0]
    
    f1=f1_score(y_real,y_hat2, average='weighted')*10000

    if args.use_demo:
        dmm = 1
    else:
        dmm = 0
    if args.time:
        tmm = 1
    else:
        tmm = 0
    results = np.concatenate([[[nm]],[[case_drop*100]],[[dmm]],[[tmm]],[[auc]],[[TN]],[[TP]],[[FN]],[[FP]],[[f1]]], axis = 1)



    with open("/data/projects/project_avabalas/RNN/results/baseline_100k.csv", "a") as myfile:
        np.savetxt(myfile, results.astype(int), fmt='%i', delimiter=',', newline='\n')
```

    EHR_RNN(
      (embed): Embedding(8600, 128, padding_idx=0)
      (rnn_c): GRU(129, 128, num_layers=2, batch_first=True, dropout=0.5)
      (out): Linear(in_features=220, out_features=1, bias=True)
      (sigmoid): Sigmoid()
    )
    case control ratio =   0.020552915158374502
    [32m
    Sample data lengths for train, test and validation:[0m
    699999 200000 100001
    [32m
     Epoch (0): Train_auc (0.9476958041172421), Valid_auc (0.946959253527227) ,Training Average_loss (0.013279137575865855), Train_time (16m 27s), Eval_time (17m 18s)[0m



    [[169512  26460]
     [   432   3596]]
    [33m
     Test_AUC (0.9484943197152519) , Test_eval_time (4m 35s) [0m
    [32mBestValidAuc 0.946959 has a TestAuc of 0.948494 at epoch 0 [0m
    [32mDetails see ../models/dhf.trainEHRmodel.log[0m



```python
from sklearn.metrics import f1_score
f1_score(y_real,y_hat2, average='weighted')
```




    0.9120975606275673




```python
# some performance measures and causes of death 
TN = cf[0][0]
TP = cf[1][1]
FP = cf[0][1]
FN = cf[1][0]
print('specificity',TN/(TN+FP))
print('sensitivity',TP/(TP+FN))
print('accuracy',(TP+TN)/(TP+FP+TN+FN))

#analysing who gets missclaissfied
import pandas as pd
res = pd.DataFrame(
    {"FINREGISTRYID": idds,
     "y_real": y_real,
     "y_hat": y_hat
    })
res.loc[(res['y_hat']>0.5)&(res['y_real']==1),'CM'] = "TP"
res.loc[(res['y_hat']>0.5)&(res['y_real']==0),'CM'] = "FP"
res.loc[(res['y_hat']<0.5)&(res['y_real']==1),'CM'] = "FN"
res.loc[(res['y_hat']<0.5)&(res['y_real']==0),'CM'] = "TN"
res['CM'].value_counts()
cod = pd.read_csv('/data/processed_data/sf_death/thl2019_1776_ksyy_tutkimus.csv.finreg_IDsp')
cod.rename(columns={'TNRO': 'FINREGISTRYID'}, inplace=True)
res = res.merge(cod, on='FINREGISTRYID', how='left')


```

    specificity 0.8649807115302186
    sensitivity 0.8927507447864945
    accuracy 0.86554
