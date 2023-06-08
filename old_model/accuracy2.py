from models.pspnet import PSPNet
import torch
from dataloader_utils2 import get_dataloader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import copy


colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0)]

pspnet = PSPNet(4, 8)
pspnet.load_state_dict(torch.load('ep041-loss0.247-val_loss0.264.pth'))
pspnet.eval()
pspnet.cuda()
train, test = get_dataloader('./LMC/lmc_askap_aconf.fits')

y_count = 0
output_count = 0
count = 0
with tqdm(total=470) as pbar:
    for xs, ys in test:
        for y in ys:
            if y.__contains__(1) or y.__contains__(2) or y.__contains__(3):
                y_count += 1
        for i in range(8):
            count += 1
            x = torch.squeeze(xs[i], 0)

            x = torch.stack([x, x, x], 2)
            img = x.numpy()
            img = img.astype(np.float32) * 255
            
            x = torch.permute(x, [2, 0, 1])
            x = torch.unsqueeze(x, 0)
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            
            output = pspnet(x)[1][0]
            output = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().detach().numpy()
            output = output.argmax(axis=-1)
            seg_img = np.zeros((np.shape(output)[0], np.shape(output)[1], 3))
            for c in range(4):
                seg_img[:, :, 0] += ((output[:, :] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((output[:, :] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((output[:, :] == c) * (colors[c][2])).astype('uint8')
            img = cv2.addWeighted(img.astype(np.float64), 0.2, seg_img, 0.8, 0)
            if output.max() > 0:
                output_count += 1
                cv2.imwrite('imgs/pos/' + str(count) + '_test.jpg', img * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                cv2.imwrite('imgs/neg/' + str(count) + '_test.jpg', img * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            pbar.update(1)
        # break
print(y_count)
print(output_count)
