import argparse
import os

import cv2
import numpy as np
from PIL import Image
from datasets import ImageDataset
from unet import *
import jittor.transform as transform


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="./", help="input_path")
parser.add_argument("--output_path", type=str, default="./results1", help="output_path")
opt = parser.parse_args()


generator = GlobalGenerator(3, 3)
discriminator = Discriminator()

generator.load(f"./results/flickr/saved_models/generator_240.pkl")
discriminator.load(f"./results/flickr/saved_models/discriminator_240.pkl")


def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

transforms = [
    transform.Resize(size=(384, 512), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

val_dataloader = ImageDataset(opt.input_path, mode="val", transforms=transforms).set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=0,
)

os.makedirs(f"{opt.output_path}", exist_ok=True)
for i, (_, real_A, photo_id) in enumerate(val_dataloader):
    print(i,"began")
    fake_B = generator(real_A)

    fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
    for idx in range(fake_B.shape[0]):
        cv2.imwrite(f"{opt.output_path}/{photo_id[idx][11:]}.jpg", fake_B[idx].transpose(1, 2, 0)[:, :, ::-1])

    print(i,"end")




