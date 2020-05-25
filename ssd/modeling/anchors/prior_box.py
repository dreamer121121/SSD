from itertools import product

import torch
from math import sqrt


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE #300
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES#[30, 60, 111, 162, 213, 264]
        self.max_sizes = prior_config.MAX_SIZES#[60, 111, 162, 213, 264, 315]
        self.strides = prior_config.STRIDES #[8, 16, 32, 64, 100, 300]
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps): #38,19,10,5,3,1
            #一个featuremap一个featuremap进行处理
            #self.strides = [8, 16, 32, 64, 100, 300]
            scale = self.image_size / self.strides[k] #计算特征图的大小,image_size=300
            for i, j in product(range(f), repeat=2):
                # unit center x,y
                # scale = image_size/stride
                # cx = cx/scale = cx * stride / image_size
                # cx * image_size = cx * stride (在原始图像上anchor的中心位置)
                cx = (j + 0.5) / scale #此时求得的cx是相对于原图的比例，用cx*image_size就是真实的anchor
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k] #[30,60,111,162,213,264]
                h = w = size / self.image_size #将anchor的w转换为相对于原图的比例,h * image_size 即得到真实的anchor的W和H
                priors.append([cx, cy, w, h])

                # big sized square box，self.max_sizes = [60, 111, 162, 213, 264, 315]
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors

if __name__ == '__main__':
    from ssd.config import cfg
    anchors = PriorBox(cfg) #8732个(38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4)
    #生成的anchor是相对于原图的比例，需要乘以原图的尺寸，得到所有anchor的坐标以(cx,cy,w,h)
    #表示，前10个：
    print(anchors()[:10]*300)
    # [[4.0000, 4.0000, 30.0000, 30.0000],
    #      [4.0000, 4.0000, 42.4264, 42.4264],
    #      [4.0000, 4.0000, 42.4264, 21.2132],
    #      [4.0000, 4.0000, 21.2132, 42.4264],
    #      [12.0000, 4.0000, 30.0000, 30.0000],
    #      [12.0000, 4.0000, 42.4264, 42.4264],
    #      [12.0000, 4.0000, 42.4264, 21.2132],
    #      [12.0000, 4.0000, 21.2132, 42.4264],
    #      [20.0000, 4.0000, 30.0000, 30.0000],
    #      [20.0000, 4.0000, 42.4264, 42.4264]]


