from utils.function import *
from torch.nn.modules.loss import _Loss

Psize = 3

#===================================ETT loss========================================
class CM(_Loss):
    __name__ = 'CM'
    def __init__(self):
      super(CM,self).__init__()
      return
    def forward(self,p,t):
      total = 0
      batch_size = t.shape[0]
      t = t.squeeze(1)
      p = p.squeeze(1)
      for t, p in zip(t, p):  
        #p = denoise(p) #Denoise
        a = find_point(p.cpu().numpy())
        b = find_point(t.cpu().numpy())
        total += (a-b).abs().mean()*Psize/72
        #print(total)
      return total/batch_size

#0.5cm
class YN(_Loss):
    __name__ = 'YN'
    def __init__(self):
      super(YN,self).__init__()
      return
    def forward(self,p,t):
      total = 0
      batch_size = t.shape[0]
      t = t.squeeze(1)
      p = p.squeeze(1)
      for t, p in zip(t, p):  
        #p = denoise(p) #Denoise
        a = find_point(p.cpu().numpy())
        b = find_point(t.cpu().numpy())
        total += ((a-b).abs()<=(72/2/Psize)).sum()
        #print(total)
      #print(a,b)
      return total/batch_size

#1cm        
class YN2(_Loss):
    __name__ = 'YN2'
    def __init__(self):
      super(YN2,self).__init__()
      return
    def forward(self,p,t):
      total = 0
      batch_size = t.shape[0]
      t = t.squeeze(1)
      p = p.squeeze(1)
      for t, p in zip(t, p):  
        #p = denoise(p) #Denoise
        a = find_point(p.cpu().numpy())
        b = find_point(t.cpu().numpy())
        total += ((a-b).abs()<=(72/Psize)).sum()
        #print(total)
      #print(a,b)
      return total/batch_size

#===================================Carina loss========================================
class CM2(_Loss):
    __name__ = 'CM2'
    def __init__(self):
      super(CM2,self).__init__()
      return
    def forward(self,p,t):
      total = 0
      batch_size = t.shape[0]
      t = t.squeeze(1)
      p = p.squeeze(1)
      for t, p in zip(t, p):  
        a = get_carina_point(p.cpu().numpy())
        b = get_carina_point(t.cpu().numpy())
        if a == None or b == None:
          total += torch.tensor(0).float()
        else:
          total += (a-b).abs().mean()*Psize/72
        #print(total)
      return total/batch_size

#0.5cm
class YN3(_Loss):
    __name__ = 'YN3'
    def __init__(self):
      super(YN3,self).__init__()
      return
    def forward(self,p,t):
      total = 0
      batch_size = t.shape[0]
      t = t.squeeze(1)
      p = p.squeeze(1)
      for t, p in zip(t, p):  
        a = get_carina_point(p.cpu().numpy())
        b = get_carina_point(t.cpu().numpy())
        if a == None or b == None:
          total += torch.tensor(0).float()
        else:
          total += ((a-b).abs()<=(72/2/Psize)).sum()
        #print(total)
      #print(a,b)
      return total/batch_size

#1cm  
class YN4(_Loss):
    __name__ = 'YN4'
    def __init__(self):
      super(YN4,self).__init__()
      return
    def forward(self,p,t):
      total = 0
      batch_size = t.shape[0]
      t = t.squeeze(1)
      p = p.squeeze(1)
      for t, p in zip(t, p):  
        a = get_carina_point(p.cpu().numpy())
        b = get_carina_point(t.cpu().numpy())
        if a == None or b == None:
          total += torch.tensor(0).float()
        else:
          total += ((a-b).abs()<=(72/Psize)).sum()
        #print(total)
      #print(a,b)
      return total/batch_size