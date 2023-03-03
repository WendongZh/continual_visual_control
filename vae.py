import torch
import torch.nn as nn
from torch.optim import Adam


class VAE(nn.Module):

  def __init__(self, num_classes=4, z_dim=128, img_width=64, lr=1e-4, kl_weight=1e-4):
    super(VAE, self).__init__()
    self.Prior = Prior_FP(num_classes=num_classes, z_dim=z_dim*2)
    self.Dec = head_reconstructor(num_classes=num_classes, z_dim=z_dim)
    self.Enc = shared_encoder(num_classes=num_classes, img_width=img_width)
    self.Latent = latent_encoder(z_dim=z_dim)
    self.lr = lr
    self.kl_weight = kl_weight

    label_tmp = torch.eye(num_classes).unsqueeze(0)
    # shape of label_tmp is 1, num_classes, num_classes
    self.label_use = nn.Parameter(data=label_tmp, requires_grad=False)

    self.VAE_opt = Adam(
      [{'params': self.Prior.parameters(), 'lr': float(self.lr)},
       {'params': self.Dec.parameters(), 'lr': float(self.lr)},
       {'params': self.Enc.parameters(), 'lr': float(self.lr)},
       {'params': self.Latent.parameters(), 'lr': float(self.lr)},
      ],
      lr=float(self.lr),
    )

    self.MSE_criterion = nn.MSELoss()

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

  def kl_criterion(self, mu1, logvar1, mu2, logvar2):
    bs, c = mu1.size()
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / bs

  def train(self, img, label):
    # img: B, 3, H, W   label: int
    # the img should be in range(0, 1)

    # make label
    self.Dec.train()
    self.Enc.train()
    img = img.permute(0, 3, 1, 2)
    # # assert img.min() >= 0 and img.max() <= 1
    # if img.min() < 0 or img.max() >1 :
    #   print([img.min(), img.max()])
    img = torch.clip(img, 0.0, 1.0)
    bs, _, _, _ = img.size()
    label = self.label_use[:, label]
    label = label.repeat(bs, 1).detach()

    mu_prior, logvar_prior = self.Prior(label)
    img_embed = self.Enc(img, label)
    mu, logvar = self.Latent(img_embed)
    loss_kl = self.kl_criterion(mu, logvar, mu_prior, logvar_prior)
    z_recon = self.reparameterize(mu, logvar)
    z_recon = torch.cat([z_recon, label], dim = 1)
    img_recon = self.Dec(z_recon)
    loss_recon = self.MSE_criterion(img_recon, img)

    loss_all = loss_recon + self.kl_weight * loss_kl

    self.VAE_opt.zero_grad()
    loss_all.backward()
    self.VAE_opt.step()

    return loss_recon.detach().cpu().numpy(), loss_kl.detach().cpu().numpy()

  def sample(self, label, batch_size, img=None):
    # label: int
    # remeber to set model.eval() before doing sampling
    self.Dec.eval()
    if img is not None:

      bs, _, _, _ = img.size()
      label = self.label_use[:, label]
      label = label.repeat(bs, 1).detach()

      img_embed = self.Enc(img, label)
      mu, logvar = self.Latent(img_embed)
      z_recon = self.reparameterize(mu, logvar)
      z_recon = torch.cat([z_recon, label], dim = 1)
      img_recon = self.Dec(z_recon)

    else:

      label = self.label_use[:, label]
      label = label.repeat(batch_size, 1).detach()

      mu_prior, logvar_prior = self.Prior(label)
      z_recon = self.reparameterize(mu_prior, logvar_prior)
      z_recon = torch.cat([z_recon, label], dim = 1)
      img_recon = self.Dec(z_recon)
      img_recon = torch.clip(img_recon, 0.0, 1.0)

    return img_recon

class Prior_FP(nn.Module):
    def __init__(self, num_classes=4, z_dim=128):
        super(Prior_FP, self).__init__()
        self.fc1=nn.Linear(num_classes, z_dim)
        self.z_dim = z_dim
    def forward(self, y):
        out=self.fc1(y)
        mu,logvar=torch.split(out, self.z_dim // 2, dim=1)
        return mu, logvar

class head_reconstructor(nn.Module):
    def __init__(self, num_classes=4, z_dim=128):
        super(head_reconstructor, self).__init__()
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        self.decoder_input = nn.Linear(z_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        #self.decvgg=vggdecoder(configs.z_dim)
    def forward(self,z):
        result=self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class shared_encoder(nn.Module):
    def __init__(self, num_classes=4, img_width=64):
        super(shared_encoder, self).__init__()

        self.img_width = img_width
        in_channels=3
        self.embed_class = nn.Linear(num_classes, self.img_width * self.img_width)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        modules = []

        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        in_channels+=1
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        #self.encvgg=vggencoder(256 )
    def forward(self, input, y):
        #[batch, channel=1, height, width]
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_width, self.img_width).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        h1=self.encoder(x)
        return h1

class latent_encoder(nn.Module):
    def __init__(self,  z_dim=128):
        super(latent_encoder, self).__init__()
        self.fc_mu = nn.Linear(512*4, z_dim)
        self.fc_var = nn.Linear(512*4, z_dim)

    def forward(self, x_hiddens):
        result = torch.flatten(x_hiddens, start_dim=1)
        #result=self.fc1(result)
        mu=self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var
