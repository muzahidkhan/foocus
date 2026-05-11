from collections import OrderedDict

import modules.core as core
import torch
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import downloading_upscale_model

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None


def perform_upscale(img, scale=2.0):
    global model

    print(f'Upscaling image with shape {str(img.shape)} to {scale}x ...')

    if model is None:
        model_filename = downloading_upscale_model()
        sd = torch.load(model_filename, weights_only=True)
        sdo = OrderedDict()
        for k, v in sd.items():
            sdo[k.replace('residual_block_', 'RDB')] = v
        del sd
        model = ESRGAN(sdo)
        model.cpu()
        model.eval()

    current_scale = 1.0
    while current_scale < scale:
        img = core.numpy_to_pytorch(img)
        img = opImageUpscaleWithModel.upscale(model, img)[0]
        img = core.pytorch_to_numpy(img)[0]
        current_scale *= 2.0

    return img
