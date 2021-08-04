#  %%
#  $ make
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import NPC8nodes
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import convolve
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %% Cheap and cheerful PSF

scale = 4.0
psf_w = 16
psf_h = 16
# scale = 1.0
# int(12/scale)
static_psf = (
    np.ones((int(12 / scale), int(12 / scale))) / int(12 / scale) ** 2
)  # Boxcar


def psf_guass(w=psf_w, h=psf_h, sigma=3):
    # blank_psf = np.zeros((w,h))
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    return gaussian(xx, 0, sigma) * gaussian(yy, 0, sigma)


static_psf = psf_guass(w=psf_w, h=psf_h, sigma=1 / 5)
plt.imshow(static_psf)
# %% Make distoplot
#

from sklearn.metrics.pairwise import euclidean_distances


def distogram(list_of_nodes, normaliser=128):
    # 128 normalises within the image,
    return euclidean_distances(list_of_nodes, list_of_nodes) / normaliser


# %% Helper functions to wrap NPC import
### Parameters
def get_npc():
    symmet = 8  # Rotational symmetry of the NPC
    mag = 30  # Magnitude of deformation [nm]; 3 standard deviation -> 99.7 % of forces on a node lie within this range
    nConnect = 2  # Number of connected neighbour nodes in clock-wise and anti-clockwise direction
    nRings = 2  # Number of rings
    r = [50, 54]
    ringAngles = [0, 0.2069]
    z = [0, 0]

    deformNPC = NPC8nodes.DeformNPC(
        nConnect, mag, symmet=symmet, nRings=nRings, r=r, ringAngles=ringAngles, z=z
    )
    ### Instantiate DeformNPC
    solution = deformNPC.solution
    fcoords = deformNPC.fcoords  # coordinates of force vectors
    initcoords = deformNPC.initcoords  # starting coordinates
    randfs = deformNPC.randfs  # magnitude of forces
    z = deformNPC.z
    r = deformNPC.r

    # NPC8nodes.XYoverTime(deformNPC.solution)
    # NPC8nodes.Plot2D(deformNPC.solution, anchorsprings=False, radialsprings=False, trajectory=False)
    # Export2CSV

    pos2D_ring_1 = NPC8nodes.Pos2D(solution[0])
    pos2D_ring_2 = NPC8nodes.Pos2D(solution[1])

    final_position = np.concatenate([pos2D_ring_1[-1], pos2D_ring_2[-1]])
    final_position
    return final_position


def npc_image_from_position(final_position, image_size=[128, 128]):
    cropped_image = np.zeros(image_size)

    final_position_int = ((final_position + image_size) / 2).astype(int)
    final_position_int_x = (final_position_int[:, 0]).astype(int)
    final_position_int_y = (final_position_int[:, 1]).astype(int)

    cropped_image[final_position_int_x, final_position_int_y] = 1
    return convolve(cropped_image, static_psf)
    # return cropped_image


#  %% Print image for debugging
final_position = get_npc()
npc_image = npc_image_from_position(final_position)
plt.imshow(npc_image)


#  %%
dist = euclidean_distances(final_position)
sns.heatmap(dist)
# %%

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#  %% Build model (first attempt, bit crap)
#   Currently a pair of unets "glued together" with a fully connected layer
#   Probably sufficient given enough data

# class Net(nn.Module):

#     def __init__(self,batch_size=16,n_class=1):
#         super(Net, self).__init__()
#         self.batch_size = batch_size
#         # self.conv1 = nn.Conv2d(1, 32, 3, 3)
#         # self.conv2 = nn.Conv2d(32, 64, 3, 3)
#         # self.dropout1 = nn.Dropout(0.25)
#         # self.dropout2 = nn.Dropout(0.5)
#         # self.fc1 = nn.Linear(9216, 128)
#         # self.fc2 = nn.Linear(128, 10)

#         # self.conv1 = nn.Conv2d(1, 1, 1, 1)
#         # self.dp = nn.Dropout(0.5)

#         self.conv1 = nn.Conv2d(1, 1, 1, 1)
#         self.dp = nn.Dropout(0.5)
#         self.conv2 = nn.Conv2d(1, 1, 1, 1)
#         self.conv3 = nn.Conv2d(1, 1, 5, 5)
#         self.fc2 = nn.AdaptiveAvgPool2d((100,100))
#         self.fc1 = nn.AdaptiveAvgPool2d((16,16))
#         self.conv4 = nn.Conv2d(1, 1, 1, 1)


#     def forward(self, x):
#         # x = x.double()
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.dp(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv3(x)
#         x = F.relu(x)
#         # x = x.view((1,-1))
#         x = self.fc1(x)
#         x = F.relu(x)
#         # x = x.view((self.batch_size,1,14,14))
#         x = self.conv4(x)
#         output = x

# # def double_conv(in_channels, out_channels):
# #     return nn.Sequential(
# #         nn.Conv2d(in_channels, out_channels, 3, padding=1),
# #         nn.ReLU(inplace=True),
# #         nn.Conv2d(out_channels, out_channels, 3, padding=1),
# #         nn.ReLU(inplace=True)
# #     )

#  %% Build model
#   Currently a pair of unets "glued together" with a fully connected layer
#   Probably sufficient given enough data

import torchvision

#  Unet structure outright stolen from the internet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    # Call is essentially the same as running "forward"
    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand


class Net(nn.Module):
    def __init__(self, batch_size=16, n_class=1):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # self.conv1 = nn.Conv2d(1, 32, 3, 3)
        # self.conv2 = nn.Conv2d(32, 64, 3, 3)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)

        # self.conv1 = nn.Conv2d(1, 1, 1, 1)
        # self.dp = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(1, 1, 1, 1)
        self.dp = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
        self.conv3 = nn.Conv2d(1, 1, 5, 5)
        self.fc2 = nn.AdaptiveAvgPool2d((100, 100))
        self.fc1 = nn.AdaptiveAvgPool2d((16, 16))
        self.conv4 = nn.Conv2d(1, 1, 1, 1)
        self.unet = UNet(1, 1)

    def forward(self, x):
        # x = x.double()
        x = self.unet(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.dp(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv3(x)
        # x = F.relu(x)
        # # # x = x.view((1,-1))
        x = self.fc1(x)

        # x = F.relu(x)
        # # x = x.view((1,-1))
        # x = self.fc1(x)
        x = self.unet(x)
        # x = F.relu(x)
        # x = x.view((self.batch_size,1,14,14))
        # x = self.conv4(x)
        # x = torch.sigmoid(x)
        # x = F.relu(x)
        output = x
        return output


#  Test Unet is working correctly by feeding in random data with the correct dimensionality
#  One channel in and out
#  You can also use channels to encode time which is useful
unet = UNet(1, 1)
x = torch.randn(16, 1, 128, 128)
unet(x).shape

#  Testing full model
model = Net(1, 1)
x = torch.randn(16, 1, 128, 128)
model(x).shape


# %% Define model and push to gpu (if availiable)
model = Net(1, 1).to(device)
#  Test
x = torch.randn(16, 1, 128, 128).to(device)

output = model(x)
output.shape


#  %% Dataload
#  Custom dataloader, simply calls the NPC generator
#  Crucially it can be
from torch.utils.data import Dataset, DataLoader


class NPCDataSet(Dataset):
    def __init__(self, samples):
        super(NPCDataSet, self).__init__()
        self.samples = samples
        # assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size

    def get_data():
        final_position = get_npc()
        npc_image = npc_image_from_position(final_position)
        distogram = euclidean_distances(final_position)
        return npc_image, distogram

    def __len__(self):
        return self.samples

    #  Usually requires an index for limited bounds data,
    #  but we have infinite data due to simulation
    def __getitem__(self, index):
        # npc_image, distogram = self.get_data()
        final_position = get_npc()
        npc_image = npc_image_from_position(final_position)
        distogram = euclidean_distances(final_position) / 128
        return npc_image, distogram


#  %% 
samples = 1000 # Number of samples per epoch, any number will do
dataset = NPCDataSet(samples) # Generate callable datatset (it's an iterator)
batch_size = 16 # Publications claim that smaller is better for batch
# model = Net(batch_size).to(device)#
# %% Define model and push to gpu (if availiable)

model = Net(1, 1).to(device)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
i = 0 
# Make a run output folder for convenience
writer = SummaryWriter()
# %%
# Always set a manual seed else your runs will be incomparable
torch.manual_seed(42)

# batch_size = 16
lr = 1e-2 # Low learning rates converge better
log_interval = 10 # Logging interval
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam does better than SGD
model.train() #Set model to training mode (probably deprecated?)
# loss_fn = nn.MSELoss(reduction='mean')
loss_fn = nn.MSELoss() # MSE is fine for this

# loss_fn = nn.BCELoss()
# loss_fn = nn.BCEWithLogitsLoss()


for epoch in range(0, epochs):
    for batch_idx, (inputs, outputs) in enumerate(loader):
        # data, target = data.to(device), target.to(device)
        final_position = get_npc()
        npc_image = npc_image_from_position(final_position)
        distogram = euclidean_distances(final_position)

        # inputs = npc_image
        # outputs = distogram
        # inputs.shape[-2:0]
        # data = torch.tensor(inputs).view([-1,1,*(inputs.shape)]).float().to(device)
        # target = torch.tensor(outputs).view([-1,1,*(outputs.shape)]).float().to(device)
        data = inputs.view([-1, 1, 128, 128]).float().to(device)
        target = outputs.view([-1, 1, 16, 16]).float().to(device)

        optimizer.zero_grad()

        for dwell in range(0, 1):
            output = model(data)
            loss = loss_fn(output, target)
            # loss = torch.nn.functional.nll_loss( output, target)
            loss.backward()
            optimizer.step()
        i += 1
        writer.add_scalar("Loss/train", loss, i)
        writer.add_image("output", output[0].view([1, 16, 16]))
        writer.add_image("target", target[0].view([1, 16, 16]))
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {str(epoch)} {str(batch_idx)} \t Loss: {str(loss.item())}"
            )
#  %%
plt.imshow(output.cpu().detach()[0, 0, :, :])
#  %%
plt.imshow(target.cpu().detach()[0, 0, :, :])

#  %%

plt.imshow(inputs.cpu().detach()[0, :, :])


# %%
# # %%
# torch.manual_seed(42)

# batch_size = 1
# lr = 1e-1
# log_interval = 10
# epochs = 100
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# model.train()
# samples = 100
# for epoch in range(0,epochs):
#     for batch_idx in range(0,samples):
#         # data, target = data.to(device), target.to(device)
#         final_position = get_npc()
#         npc_image = npc_image_from_position(final_position)
#         distogram = euclidean_distances(final_position)

#         data = torch.tensor(npc_image).view([1,1,*(npc_image.shape)]).float().to(device)
#         target = torch.tensor(distogram).view([1,1,*(distogram.shape)]).float().to(device)

#         optimizer.zero_grad()
#         output = model(data)

#         loss = nn.MSELoss()( output.view((1,-1)), target.view((1,-1)))
#         loss.backward()
#         optimizer.step()

#         if batch_idx % log_interval == 0:
#             print(f'Train Epoch: {str(epoch)} {str(batch_idx)} \t Loss: {str(loss.item())}')

# %%
# image_size = 2048
# npc_num = 100
# centres = np.random.randint(0,image_size,size=[npc_num,2])

# for i,centre in enumerate(centres):
#     final_position+centre