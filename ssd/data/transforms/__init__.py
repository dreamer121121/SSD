from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(), #将image的数据数据类型转换为np.float32
            PhotometricDistort(),#准换颜色模型，RGB，HSV等
            # 将image进行扩展（补0），image.size相对于原始的图片变大了，同时GT_box的坐标也随之改变
            Expand(cfg.INPUT.PIXEL_MEAN),
            #对扩展后的图片进行随机剪裁，image.size会被改变
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE), #转换为图像的尺寸为300X300
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    #给anchor 打标签，利用anchor和GT
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
