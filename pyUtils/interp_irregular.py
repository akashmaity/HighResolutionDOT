import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

def grid_contrib(input, grid):
    '''
    Interpolation over an irregular grid

    inputs:
    input - input image value tensor: NCHW format.  The image intensity values at irregular grid samples
    grid - input grid locations: NHW2 format. 

    outputs:
    interpolated images, the image intensities are defined over regular grids

    '''
    N, C, H, W = input.shape
    x = (grid[:, :, :, 0] + 1) * (W - 1) / 2
    y = (grid[:, :, :, 1] + 1) * (H - 1) / 2
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0_c = torch.clamp(x0, 0, W - 1)
    x1_c = torch.clamp(x1, 0, W - 1)
    y0_c = torch.clamp(y0, 0, H - 1)
    y1_c = torch.clamp(y1, 0, H - 1)

    base = (torch.arange(N) * H * W).unsqueeze(1).expand((N, H * W)).view(-1).long().to(input.device)
    base_y0 = base + y0_c.view(-1) * W
    base_y1 = base + y1_c.view(-1) * W
    idx_a = base_y0 + x0_c.view(-1)
    idx_b = base_y1 + x0_c.view(-1)
    idx_c = base_y0 + x1_c.view(-1)
    idx_d = base_y1 + x1_c.view(-1)
    input_trans = input.transpose(1, 2).transpose(2, 3)

    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()
    wa = (x1_f - x.float()) * (y1_f - y.float())
    wb = (x1_f - x.float()) * (y.float() - y0_f)
    wc = (x.float() - x0_f) * (y1_f - y.float())
    wd = (x.float() - x0_f) * (y.float() - y0_f)
    wa = ((x0_c == x0) * (y0_c == y0)).float() * wa
    wb = ((x0_c == x0) * (y1_c == y1)).float() * wb
    wc = ((x1_c == x1) * (y0_c == y0)).float() * wc
    wd = ((x1_c == x1) * (y1_c == y1)).float() * wd
    output = torch.zeros(N * H * W, C).to(input.device)
    idx_a = idx_a.unsqueeze(1) + output.long()
    idx_b = idx_b.unsqueeze(1) + output.long()
    idx_c = idx_c.unsqueeze(1) + output.long()
    idx_d = idx_d.unsqueeze(1) + output.long()
    output = output.scatter_add_(0, idx_a, (input_trans * wa.unsqueeze(3)).reshape(-1, C))
    output = output.scatter_add_(0, idx_b, (input_trans * wb.unsqueeze(3)).reshape(-1, C))
    output = output.scatter_add_(0, idx_c, (input_trans * wc.unsqueeze(3)).reshape(-1, C))
    output = output.scatter_add_(0, idx_d, (input_trans * wd.unsqueeze(3)).reshape(-1, C))
    output = output.reshape(N, H, W, C).transpose(2, 3).transpose(1, 2)
    return output


if __name__ == '__main__':
    im = torch.ones((1, 1, 8, 8))
    u, v = np.meshgrid(np.linspace(-1, 1, im.shape[3]), np.linspace(-1, 1, im.shape[2]))
    v = torch.tensor(v)
    u = torch.tensor(u)
    grid = torch.stack((u, v), dim=2).expand((im.shape[0], im.shape[2], im.shape[3], 2)) # NHW2
    grid = grid + 0.1
    contrib = grid_contrib(im, grid)
    print(contrib[0,0,:,:])
    plt.imshow(  contrib[0, 0, :, : ].cpu().numpy() )
    plt.show()
