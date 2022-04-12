import torch.nn.functional as F
import torch.optim as optim

# lose_fun=F.cross_entropy(1,2)
optimizer=optim.SGD
print(type(optimizer))