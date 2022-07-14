import os
from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
# %matplotlib inline

import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = os.path.join(os.getcwd(), "data")


transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((256,256))
])
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

# check
x = torch.randn(16, 3, 256,256, device=device)
model = UNetDown(3,64).to(device)
down_out = model(x)
#print(down_out.shape)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

# check
x = torch.randn(16, 128, 64, 64, device=device)
model = UNetUp(128,64).to(device)
out = model(x,down_out)
#print(out.shape)

class DestDataset(Dataset):
    def __init__(self, path2img, direction='b2a', transform=False):
        super().__init__()
        self.direction = direction
        self.path2a = join(path2img, 'a')
        self.path2b = join(path2img, 'b')
        self.img_filenames = [x for x in listdir(self.path2a)]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2a, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2b, self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        if self.direction == 'b2a':
            return b, a
        else:
            return a, b

    def __len__(self):
        return len(self.img_filenames)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)
        self.down3 = UNetDown(128,256)
        self.down4 = UNetDown(256,512,dropout=0.5)
        self.down5 = UNetDown(512,512,dropout=0.5)
        self.down6 = UNetDown(512,512,dropout=0.5)
        self.down7 = UNetDown(512,512,dropout=0.5)
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8

# check
x = torch.randn(16,3,256,256,device=device)
model = GeneratorUNet().to(device)
out = model(x)
#print(out.shape)


class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


# check
x = torch.randn(16, 64, 128, 128, device=device)
model = Dis_block(64, 128).to(device)
out = model(x)
#print(out.shape)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x
# check
x = torch.randn(16,3,256,256,device=device)
model = Discriminator().to(device)
out = model(x,x)
#print(out.shape)

model_gen = GeneratorUNet().to(device)
model_dis = Discriminator().to(device)


# 가중치 불러오기
path2models = root + '/models/num/5/'

path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

weights = torch.load(path2weights_gen,map_location=torch.device('cpu'))
model_gen.load_state_dict(weights)

# evaluation model
model_gen.eval()


train_ds = DestDataset(root, transform=transform)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

root_test = os.path.join(os.getcwd(), "root_test")

test_ds = DestDataset(root_test, transform=transform)

test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)
#test_dl

# 가짜 이미지 생성
with torch.no_grad():
    for a,b in train_dl:
        fake_imgs = model_gen(a.to(device)).detach().cpu()
        draw_imgs = a
        real_imgs = b
        break
#real_imgs.shape

# 가짜 이미지 시각화
#plt.figure(figsize=(20,20))

for i in range(0,32,3):
    plt.subplot(12,6,i+1)
    #plt.imshow(to_pil_image(0.5*real_imgs[i]+0.5))
    plt.axis('off')
    plt.subplot(12,6,i+2)
    #plt.imshow(to_pil_image(0.5*draw_imgs[i]+0.5))
    plt.axis('off')
    plt.subplot(12,6,i+3)
    #plt.imshow(to_pil_image(0.5*fake_imgs[i]+0.5))
    plt.axis('off')



a,b = test_ds[0]
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
#plt.imshow(to_pil_image(0.5*a+0.5))
plt.axis('off')
plt.subplot(1,2,2)
#plt.imshow(to_pil_image(0.5*b+0.5))
plt.axis('off')



# evaluation model
test_model_gen = model_gen.eval()

with torch.no_grad():
    for a,b in test_dl:
        test_fake_imgs = test_model_gen(a.to(device)).detach().cpu()
        test_draw_imgs = b
        break
#real_imgs.shape

# 가짜 이미지 시각화
#plt.figure(figsize=(20,20))

for i in range(0,2,2):
    plt.subplot(12,6,i+1)
    #plt.imshow(to_pil_image(0.5*test_draw_imgs[i]+0.5))
    #plt.show()
    plt.axis('off')
    plt.subplot(12,6,i+2)
    #plt.imshow(to_pil_image(0.5*test_fake_imgs[i]+0.5))
    #plt.show()
    to_pil_image(0.5*test_fake_imgs[i]+0.5).save("./fake_images/fake_img.png",'png')
    to_pil_image(0.5 * test_fake_imgs[i] + 0.5).save("./static/user_fake_images/5/fake_img.png",'png')
    plt.axis('off')

def re(a):
    return a
re(to_pil_image(0.5*test_fake_imgs[0]+0.5))
#print(to_pil_image(0.5*test_fake_imgs[0]+0.5))
