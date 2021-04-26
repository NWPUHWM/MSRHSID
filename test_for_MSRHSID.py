
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

from psnr import mpsnr
from psnr import mssim
torch.cuda.set_device(0)
#The MSRHSID model
class predict_noise_map(nn.Module):
    def __init__(self):
        super(predict_noise_map, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.ReLU()
        )
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)

        return x
class BCU(nn.Module):
    def __init__(self, in_channels):
        super(BCU, self).__init__()
        self.bcu = nn.Sequential(

            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(),

        )

    def forward(self, x):
        out = self.bcu(
            F.interpolate(x, size=(int(x.size(2)) // 2, int(x.size(3)) // 2), mode='bilinear', align_corners=True))
        out = F.relu(F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=True))
        return out
class MRM(nn.Module):
    def __init__(self, in_channels):
        super(MRM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU()
        )

        self.bcu = BCU(in_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        bcu = self.bcu(conv1)
        conv3 = self.conv3(conv2 + bcu)
        return conv3 + x


class BlindHSID(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(BlindHSID, self).__init__()
        self.f2_3 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, 3, 1, 1),
            nn.ReLU(),

        )
        self.f3_3 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, 3, 1, 1),
            nn.ReLU(),
        )

        self.MRMS = nn.Sequential(
            MRM(out_channels),
            MRM(out_channels),
            MRM(out_channels),
            MRM(out_channels),

        )

        self.out = nn.Sequential(
            nn.Conv2d(out_channels, 1, 3, 1, 1),

        )
        self._initialize_weights()

    def forward(self, x, y, x_map):
        f2_3 = self.f2_3(torch.cat((x, x_map), dim=1))
        f3_3 = self.f3_3(y.squeeze(1))
        mrm = self.MRMS(f2_3 + f3_3)

        out = self.out(mrm)

        return out + x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
# load the trained model of prediction and MSRHSID
def predict():
    models = predict_noise_map()
    models.load_state_dict(torch.load('predict_noise.pth', map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
def model_rand():
    models=BlindHSID(2,72,32)
    models.load_state_dict(torch.load('MSRHSID.pth',map_location='cuda:0'))
    models.cuda()
    models.eval()
    return models
# the denoising function
def teste(tests,model,predict,test,noise_map):

    cs=191
    k = 18
    tests = tests.astype(np.float32)
    noise_map_predict = np.ones(tests.shape).astype(np.float32)
    # prediction process
    for i in range(cs):

        predict_image = tests[i, :, :]
        predict_image = np.expand_dims(predict_image, axis=0)
        predict_image = np.expand_dims(predict_image, axis=0)
        predict_image = torch.from_numpy(predict_image)
        out = predict(predict_image.cuda()).data.cpu().numpy()
        # stds = (tests[i, :, :] - test[i, :, :]).std()
        # noise_map_predict[i, :, :] = noise_map_predict[i, :, :] * stds.item()
        noise_map_predict[i, :, :] = noise_map_predict[i, :, :] *out.mean()





    test_out = np.zeros(tests.shape).astype(np.float32)

    noise_map = noise_map_predict
    #denoising process
    for channel_i in range(cs):
        x = tests[channel_i, :, :]
        x_1 = noise_map[channel_i, :, :]

        x_1 = np.expand_dims(x_1, axis=0)
        x_1 = np.expand_dims(x_1, axis=0)
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        if channel_i < k:

            y = tests[0:36, :, :]
            y_1 = noise_map[0:36, :, :]

        elif channel_i < cs - k:

            y = np.concatenate((tests[channel_i - k:channel_i, :, :],
                                tests[channel_i + 1:channel_i + k + 1, :, :]))
            y_1 = np.concatenate((noise_map[channel_i - k:channel_i, :, :],
                                  noise_map[channel_i + 1:channel_i + k + 1, :, :]))

        else:
            y = tests[cs - 36:cs, :, :]
            y_1 = noise_map[cs - 36:cs, :, :]
        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=0)

        y_1 = np.expand_dims(y_1, axis=0)
        y_1 = np.expand_dims(y_1, axis=0)

        x = torch.from_numpy(x)
        x_1 = torch.from_numpy(x_1)

        y = torch.from_numpy(y)
        y_1 = torch.from_numpy(y_1)
        y=torch.cat((y,y_1),dim=2)

        with torch.no_grad():

            out = model(x.cuda(), y.cuda(), x_1.cuda())

        out = out.squeeze(0)

        out = out.data.cpu().numpy()

        test_out[channel_i, :, :] = out

    test_out=test_out.astype(np.double)

    return test_out

def main():
    SIGMA=25
    test = np.load("./test_washington.npy")
    test = test.transpose((2, 1, 0))
    tests = np.zeros(test.shape).astype(np.float32)
    noise_map=np.ones(test.shape).astype(np.float32)
    #add noise
    for i in range(191):
        # +(np.random.randint(0, 26) / 255.0) * np.random.randn(200, 200)
        tests[i, :, :] = test[i, :, :] + +(np.random.randint(10, 55) / 255.0) * np.random.randn(test.shape[2], test.shape[1])
    pre = predict()
    model = model_rand()
    test_out = teste(tests, model, pre, test, noise_map)



    test_out = test_out.transpose((2, 1, 0))
    test = test.transpose((2, 1, 0))
    print("test_shape:",test.shape,"test_out_shape:", test_out.shape)

    ssim = mssim(test, test_out)
    psnr = mpsnr(test, test_out)
    print("PSNR:",psnr,"SSIM:", ssim)

main()