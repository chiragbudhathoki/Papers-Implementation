import torch
import torchvision 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import os


class Discriminator(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self,z_dim,in_features):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,out_features = 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,in_features),#the in_features will be 28x28 i.e 784
            nn.Tanh()
        )
    def forward(self,x):
        return self.gen(x)

#hyperparameter
device = "cuda"if torch.cuda.is_available()else "cpu"
lr_disc = 1e-4
lr_gen = 3e-4
z_dim = 128
in_features = 28 * 28 * 1
batch_size = 256
epochs = 100

disc = Discriminator(in_features).to(device)
gen = Generator(z_dim,in_features).to(device)
fixed_noise = torch.randn((batch_size,z_dim))

transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize((0.1307,),(0.3081,))
])
dataset = torchvision.datasets.MNIST(root = "dataset/",transform = transform,download = True)
dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True)
optim_disc = optim.Adam(disc.parameters(),lr = lr_disc,betas = (0.5,0.999))
optim_gen = optim.Adam(gen.parameters(), lr = lr_gen,betas = (0.5,0.999))
cost = nn.BCELoss()

base_dir = os.path.dirname(os.path.abspath(__file__))

writer_fake = SummaryWriter(os.path.join(base_dir, "runs", "GAN_MNIST", "fake"))
writer_real = SummaryWriter(os.path.join(base_dir, "runs", "GAN_MNIST", "real"))

step = 0

for epoch in range(epochs):
    for batch_idx,(real,_) in enumerate(dataloader):
        real = real.view(-1,784).to(device)
        batch_size = real.shape[0]

        #training discriminator = max(log[D(real)]+log[1 - D(G(z))])
        noise = torch.randn(batch_size,z_dim).to(device)#just a random noise 
        fake = gen(noise)#generating img from the noise

        #log[D(real)]
        disc_real = disc(real).view(-1)#just taking the real img and flattening everything
        lossD_real = cost(disc_real,torch.full_like(disc_real,0.9))

        #### Training Discriminator #####
        #### max log[1-D(G(z))] ####
        disc_fake = disc(fake).view(-1) #just flattening everything
        lossD_fake = cost(disc_fake,torch.zeros_like(disc_fake))#we want the discriminator to output 0 for the fake images

        
        lossD = (lossD_real + lossD_fake)/2 #this is E(expectation) talked in the paper
        
        disc.zero_grad()
        lossD.backward(retain_graph = True)#the retain_graph is used to retain the fake as it is needed in the generator too
        
        optim_disc.step()

        ##### Training Generator ####
        #### min(Log[1-D(G(z)] but this expression leads to weak gradients which slower training or no training so we instead use max(Log[D(G(z))]) ####
        output = disc(fake).view(-1)
        lossG = cost(output,torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        optim_gen.step()

        #### TensorBoard ####
        if batch_idx == 0:
            print(f'Epoch:{epoch} ,Loss D : {lossD:.4f}, Loss G: {lossG:.4f}')
            writer_fake.add_scalar("Loss/Discriminator", lossD.item(), step)
            writer_fake.add_scalar("Loss/Generator", lossG.item(), step)
            with torch.no_grad():
                fake = gen(noise).reshape(-1,1,28,28)
                x = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake,normalize = True)
                img_grid_x = torchvision.utils.make_grid(x,normalize = True)

                writer_fake.add_image (
                    "MNIST Fake Images",img_grid_fake,global_step = step 
                )
                writer_real.add_image(
                    "MNIST Real Images",img_grid_x , global_step = step
                )

        step += 1
    