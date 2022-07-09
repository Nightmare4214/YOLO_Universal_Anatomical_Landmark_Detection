import torch
from AC_loss import AC_loss

# loss
l1 = torch.nn.L1Loss
l2 = torch.nn.MSELoss
bce = torch.nn.BCELoss
ac_loss = AC_loss

# optimizer
adam = torch.optim.Adam
sgd = torch.optim.SGD
adagrad = torch.optim.Adagrad
rmsprop = torch.optim.RMSprop

# scheduler 
steplr = torch.optim.lr_scheduler.StepLR
multisteplr = torch.optim.lr_scheduler.MultiStepLR
cosineannealinglr = torch.optim.lr_scheduler.CosineAnnealingLR
reducelronplateau = torch.optim.lr_scheduler.ReduceLROnPlateau
lambdalr = torch.optim.lr_scheduler.LambdaLR
cycliclr = torch.optim.lr_scheduler.CyclicLR
