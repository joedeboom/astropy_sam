from models.pspnet import PSPNet
import torch
from dataloader_utils2 import get_dataloader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


pspnet = PSPNet(4, 8)
pspnet.load_state_dict(torch.load('ep032-loss0.250-val_loss0.258.pth'))
pspnet.eval()
pspnet.cuda()
train, test = get_dataloader('./LMC/test.fits')

y_count = 0
output_count = 0

with tqdm(total=470) as pbar:
    for xs, ys in test:
        for y in ys:
            if y.__contains__(1) or y.__contains__(2) or y.__contains__(3):
                y_count += 1
        for x in xs:
            x = x.type(torch.FloatTensor)
            x = torch.unsqueeze(x, 0)
            
            x = x.cuda()
            
            output = pspnet(x)[1][0]
            output = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().detach().numpy()
            output = output.argmax(axis=-1)
            if output.max() > 0:
                output_count += 1
        pbar.update(1)

print(y_count)
print(output_count)
