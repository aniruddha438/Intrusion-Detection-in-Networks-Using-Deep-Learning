import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import paths_1 as cf

class HyperParams:
    def __init__(self):
        # self.epoch_num = 20
        # self.batch_size = 128
        # self.input_dim = 51
        # self.cn_hidden1_dim = 26
        # self.cn_hidden2_dim = 12
        # self.cn_hidden3_dim = 6
        # self.zc_dim = 2
        # self.en_hidden_dim = 16
        # self.mixture_dim = 4
        # self.dropout_p = 0.5
        # self.lam1 = 0.1
        # self.lam2 = 0.005
        self.input_dim = 51
        self.cn_hidden1_dim = 128
        self.cn_hidden2_dim = 64
        self.cn_hidden3_dim = 32
        self.zc_dim = 16   # previously 2 — substantially larger latent
        self.mixture_dim = 6  # increase components to 5-10
        self.en_hidden_dim = 64
        self.dropout_p = 0.2
        self.lam1 = 0.1
        self.lam2 = 0.005
        self.batch_size = 128  # okay
        self.epoch_num = 20  

hyper_params = HyperParams()

def relative_euclidean_distance(x, x_hat):
    x_ = x.unsqueeze(1)
    x_hat_ = x_hat.unsqueeze(1)
    # d1 shape: [batch_size, 1]
    d1 = torch.cdist(x_, x_hat_).squeeze(1)
    # d2 shape: [batch_size, 1]
    d2 = torch.cdist(x_, torch.zeros_like(x_)).squeeze(1)
    return d1/d2


# class CompressNet(nn.Module):
#     def __init__(self, x_dim, hidden1_dim, hidden2_dim, hidden3_dim, zc_dim):
#         super(CompressNet, self).__init__()
        

#         self.encoder_layer1 = nn.Linear(x_dim, hidden1_dim)
#         self.encoder_layer2 = nn.Linear(hidden1_dim, hidden2_dim)
#         self.encoder_layer3 = nn.Linear(hidden2_dim, hidden3_dim)
#         self.encoder_layer4 = nn.Linear(hidden3_dim, zc_dim)

#         self.decoder_layer1 = nn.Linear(zc_dim, hidden3_dim)
#         self.decoder_layer2 = nn.Linear(hidden3_dim, hidden2_dim)
#         self.decoder_layer3 = nn.Linear(hidden2_dim, hidden1_dim)
#         self.decoder_layer4 = nn.Linear(hidden1_dim, x_dim)

#     def forward(self, x):
#         h = self.encoder_layer1(x)
#         h = self.encoder_layer2(h)
#         h = self.encoder_layer3(h)
#         zc = self.encoder_layer4(h)

#         h = self.decoder_layer1(zc)
#         h = self.decoder_layer2(h)
#         h = self.decoder_layer3(h)
#         x_hat = self.decoder_layer4(h)

#         # ed shape: [batch_size, 1]
#         ed = relative_euclidean_distance(x, x_hat)
#         cos = nn.CosineSimilarity(dim=1)
#         # cosim shape: [batch_size, 1]
#         cosim = cos(x, x_hat).unsqueeze(1)
#         # z shape: [batch_size, zc_dim+2]
#         z = torch.cat((zc, ed, cosim), dim=1)
#         assert zc.shape[0] == z.shape[0]
#         assert zc.shape[1] == z.shape[1] - 2

#         return z, x_hat
    
#     # def reconstruct_error(self, x, x_hat):
#     #     e = torch.tensor(0.0)
#     #     for i in range(x.shape[0]):
#     #         e += torch.dist(x[i], x_hat[i])
#     #     return e / x.shape[0]
#     def reconstruct_error(self, x, x_hat):
#     # Compute per-sample L2 distance between x and x_hat
#      e = torch.norm(x - x_hat, dim=1)   # [batch_size]
#      return e.mean()
class CompressNet(nn.Module):
    def __init__(self, x_dim, hidden1_dim, hidden2_dim, hidden3_dim, zc_dim, dropout_p=0.2):
        super(CompressNet, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(x_dim, hidden1_dim)
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.enc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.enc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.bn3 = nn.BatchNorm1d(hidden3_dim)
        # bottleneck
        self.enc4 = nn.Linear(hidden3_dim, zc_dim)

        # Decoder (mirror)
        self.dec1 = nn.Linear(zc_dim, hidden3_dim)
        self.dbn1 = nn.BatchNorm1d(hidden3_dim)
        self.dec2 = nn.Linear(hidden3_dim, hidden2_dim)
        self.dbn2 = nn.BatchNorm1d(hidden2_dim)
        self.dec3 = nn.Linear(hidden2_dim, hidden1_dim)
        self.dbn3 = nn.BatchNorm1d(hidden1_dim)
        self.dec4 = nn.Linear(hidden1_dim, x_dim)

        self.dropout = nn.Dropout(dropout_p)
        self.act = nn.LeakyReLU(0.1)

        # optional normalization on z (after concat)
        self.z_bn = nn.BatchNorm1d(zc_dim + 2)

    def forward(self, x):
        # Encoder: Linear -> BN -> LeakyReLU -> Dropout
        h = self.enc1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.enc2(h)
        h = self.bn2(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.enc3(h)
        h = self.bn3(h)
        h = self.act(h)
        h = self.dropout(h)

        zc = self.enc4(h)   # raw latent (no activation here)

        # Decoder: mirror with activations
        h = self.dec1(zc)
        h = self.dbn1(h)
        h = self.act(h)

        h = self.dec2(h)
        h = self.dbn2(h)
        h = self.act(h)

        h = self.dec3(h)
        h = self.dbn3(h)
        h = self.act(h)

        x_hat = self.dec4(h)  # final reconstruction (no activation; input scaled before)

        # compute extra features for z
        ed = relative_euclidean_distance(x, x_hat)   # [batch,1]
        cos = nn.CosineSimilarity(dim=1)
        cosim = cos(x, x_hat).unsqueeze(1)                       # [batch,1]

        # concat zc + ed + cosim
        z = torch.cat((zc, ed, cosim), dim=1)  # [batch, zc_dim + 2]

        # normalize the combined latent vector (helps GMM estimation)
        z = self.z_bn(z)
        assert zc.shape[0] == z.shape[0]
        assert zc.shape[1] == z.shape[1] - 2

        return z, x_hat

    def reconstruct_error(self, x, x_hat):
        # L2 per-sample mean (same as you had, but stable)
        e = torch.norm(x - x_hat, dim=1)   # [batch]
        return e.mean()
class EstimateNet(nn.Module):
    def __init__(self, z_dim, hidden_dim, dropout_p, mixture_dim, lam1, lam2):
        super(EstimateNet, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.mixture_dim = mixture_dim
        self.lam1 = lam1
        self.lam2 = lam2

        self.layer1 = nn.Linear(z_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_p)
        self.layer2 = nn.Linear(hidden_dim, mixture_dim)

    def forward(self, z):
        h = self.layer1(z)
        h = torch.tanh(h)
        h = self.drop(h)
        h = self.layer2(h)
        # gamma shape: [batch_size, mixture_dim]
        gamma = F.softmax(h, dim=1)
        return gamma

    # return shape: [mixture_dim]
    def mixture_prob(self, gamma):
        n = gamma.shape[0]
        return torch.sum(gamma, dim=0) / n

    # return shape: [mixture_dim, z_dim]
    def mixture_mean(self, gamma, z):
        gamma_t = torch.t(gamma)
        miu = torch.mm(gamma_t, z)
        miu = miu / torch.sum(gamma_t, dim=1).unsqueeze(1)
        return miu

    # return shape: [mixture_dim, z_dim, z_dim]
    def mixture_covar(self, gamma, z, miu):
        cov = torch.zeros((self.mixture_dim, self.z_dim, self.z_dim))
        # z_t shape: [z_dim, batch_size]
        z_t = torch.t(z)
        for k in range(self.mixture_dim):
            miu_k = miu[k].unsqueeze(1)
            # dm shape: [z_dim, batch_size]
            dm = z_t - miu_k
            # gamma_k shape: [batch_size, batch_size]
            gamma_k = torch.diag(gamma[:, k])
            # cov_k shape: [z_dim, z_dim]
            cov_k = torch.chain_matmul(dm, gamma_k, torch.t(dm))
            cov_k = cov_k / torch.sum(gamma[:, k])
            cov[k] = cov_k
        return cov
    
    # m_prob shape: [mixture_dim]
    # m_mean shape: [mixture_dim, z_dim]
    # m_cov shape: [mixture_dim, z_dim, z_dim]
    # zi shape: [z_dim, 1]

    # loss is becoming nan in this implementation 


    # def sample_energy(self, m_prob, m_mean, m_cov, zi):
    #     e = torch.tensor(0.0)
    #     cov_eps = torch.eye(m_mean.shape[1]) * (1e-12)
    #     for k in range(self.mixture_dim):
    #         # miu_k shape: [z_dim, 1]
    #         miu_k = m_mean[k].unsqueeze(1)
    #         d_k = zi - miu_k

    #         # solve the singular covariance
    #         inv_cov = torch.inverse(m_cov[k] + cov_eps)
    #         e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k))
    #         e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * m_cov[k])))
    #         e_k = e_k * m_prob[k]
    #         e += e_k.squeeze()
    #     return - torch.log(e)
    # def sample_energy(self, m_prob, m_mean, m_cov, zi):
    #  device = zi.device  # make sure everything stays on the same device
    #  e = torch.tensor(0.0, device=device)
    #  cov_eps = torch.eye(m_mean.shape[1], device=device) * 1e-12

    #  for k in range(self.mixture_dim):
    #     # miu_k shape: [z_dim, 1]
    #     miu_k = m_mean[k].unsqueeze(1).to(device)
    #     d_k = zi - miu_k

    #     # ensure covariance is on the same device
    #     cov_k = m_cov[k].to(device) + cov_eps

    #     # solve the singular covariance
    #     inv_cov = torch.inverse(cov_k)
    #     # replace chain_matmul with multi_dot
    #     e_k = torch.exp(-0.5 * torch.linalg.multi_dot([d_k.t(), inv_cov, d_k]))
    #     e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * cov_k)))
    #     e_k = e_k * m_prob[k].to(device)
    #     e += e_k.squeeze()

    #  return -torch.log(e)



    # def energy(self, gamma, z):
    #     m_prob = self.mixture_prob(gamma)
    #     m_mean = self.mixture_mean(gamma, z)
    #     m_cov = self.mixture_covar(gamma, z, m_mean)

    #     e = torch.tensor(0.0)
    #     for i in range(z.shape[0]):
    #         zi = z[i].unsqueeze(1)
    #         ei = self.sample_energy(m_prob, m_mean, m_cov, zi)
    #         e += ei
        
    #     p = torch.tensor(0.0)
    #     for k in range(self.mixture_dim):
    #         cov_k = m_cov[k]
    #         p_k = torch.sum(1 / torch.diagonal(cov_k, 0))
    #         p += p_k
        
    #     return (self.lam1 / z.shape[0]) * e + self.lam2 * p


    # def batch_sample_energy(self, m_prob, m_mean, m_cov, z):
    # # """
    # # Vectorized computation of energies for a batch of latent vectors z.
    # # Numerically stable with log-sum-exp trick.
    # # """
    #  device = z.device
    #  N, D = z.shape
    #  K = self.mixture_dim
    #  eps = 1e-6

    # # Stabilize covariances
    #  cov_eps = torch.eye(D, device=device) * eps
    #  cov_k_list = [m_cov[k].to(device) + cov_eps for k in range(K)]
    #  inv_cov = torch.stack([torch.inverse(torch.clamp(cov_k + torch.diag(torch.clamp(torch.diagonal(cov_k,0),min=eps)), min=eps)) 
    #                        for cov_k in cov_k_list])  # K x D x D
    #  det_cov = torch.stack([torch.clamp(torch.det(cov_k), min=eps) for cov_k in cov_k_list])  # K

    # # Expand z and m_mean to broadcast: N x K x D x 1
    #  z_exp = z.unsqueeze(1).transpose(2,3)  # N x 1 x D x 1 -> N x 1 x D x 1
    #  mu_exp = m_mean.unsqueeze(0).unsqueeze(3)  # 1 x K x D x 1

    #  d = z_exp - mu_exp  # N x K x D x 1
    # # quadratic form: N x K
    #  quad = torch.matmul(torch.matmul(d.transpose(2,3), inv_cov.unsqueeze(0)), d).squeeze(-1).squeeze(-1)

    # # log-probabilities N x K
    #  log_prob = -0.5 * quad - 0.5 * torch.log(det_cov * (2*math.pi)**D).unsqueeze(0)
    #  log_prob += torch.log(torch.clamp(m_prob.to(device), min=eps).unsqueeze(0))  # add mixture weight

    # # log-sum-exp across mixture components
    #  max_log = torch.max(log_prob, dim=1, keepdim=True)[0]
    #  log_sum = max_log + torch.log(torch.sum(torch.exp(log_prob - max_log), dim=1, keepdim=True))
    #  energies = -log_sum.squeeze()  # N

    #  return energies
    def sample_energy(self, m_prob, m_mean, m_cov, zi):
    # """
    # Compute the energy (negative log-likelihood) for one latent sample zi
    # in a numerically stable way.
    # """
      device = zi.device
      e = torch.tensor(0.0, device=device)

      eps = 1e-6  # small epsilon for stability
      cov_eps = torch.eye(m_mean.shape[1], device=device) * eps

      log_terms = []

      for k in range(self.mixture_dim):
          # k-th Gaussian
          mu_k = m_mean[k].unsqueeze(1).to(device)      # [D,1]
          d_k = zi - mu_k                                # [D,1]
          cov_k = m_cov[k].to(device) + cov_eps         # stabilize covariance

          # clamp diagonal to avoid division by zero
          cov_k = cov_k + torch.diag(torch.clamp(torch.diagonal(cov_k, 0), min=eps))

          # inverse and determinant safely
          inv_cov = torch.inverse(cov_k)
          det_cov = torch.det(cov_k)
          det_cov = torch.clamp(det_cov, min=eps)

          # quadratic term
          quad = torch.linalg.multi_dot([d_k.t(), inv_cov, d_k])
          log_prob = -0.5 * quad - 0.5 * torch.log(det_cov * (2 * math.pi)**cov_k.shape[0])
          
          # add mixture weight in log-space
          log_prob += torch.log(torch.clamp(m_prob[k], min=eps))
          log_terms.append(log_prob)

      # log-sum-exp trick
      log_terms = torch.cat(log_terms, dim=0)
      max_log = torch.max(log_terms)
      sample_energy = - (max_log + torch.log(torch.sum(torch.exp(log_terms - max_log))))
      
      return sample_energy.squeeze()


    # def energy(self, gamma, z):
    # # """
    # # Vectorized energy for the batch + covariance penalty.
    # # """
    #  device = z.device
    #  m_prob = self.mixture_prob(gamma)
    #  m_mean = self.mixture_mean(gamma, z)
    #  m_cov = self.mixture_covar(gamma, z, m_mean)

    # # batch energies
    #  energies = self.sample_energy(m_prob, m_mean, m_cov, z)
    #  e = torch.sum(energies)

    # # covariance penalty
    #  p = torch.tensor(0.0, device=device)
    #  eps = 1e-6
    #  for k in range(self.mixture_dim):
    #     cov_k = m_cov[k].to(device)
    #     p += torch.sum(1.0 / torch.clamp(torch.diagonal(cov_k,0), min=eps))

    #  return (self.lam1 / z.shape[0]) * e + self.lam2 * p

    def energy(self, gamma, z):
      m_prob = self.mixture_prob(gamma)
      m_mean = self.mixture_mean(gamma, z)
      m_cov = self.mixture_covar(gamma, z, m_mean)

      e = torch.tensor(0.0, device=z.device)
      for i in range(z.shape[0]):
          zi = z[i].unsqueeze(1)
          e += self.sample_energy(m_prob, m_mean, m_cov, zi)

      p = torch.tensor(0.0, device=z.device)
      for k in range(self.mixture_dim):
          cov_k = m_cov[k]
          p += torch.sum(1 / torch.diagonal(cov_k, 0))

      return (self.lam1 / z.shape[0]) * e + self.lam2 * p


