import torch


def get_fixed_coords(Ln1, Ln2):
    rows = torch.linspace(0, 1, steps=Ln1 + 1)[0:-1]
    cols = torch.linspace(0, 1, steps=Ln2 + 1)[0:-1]
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.cat([rr.reshape(-1, 1), cc.reshape(-1, 1)], dim=-1)
    return coords


class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError()


class RBFKernel(NoiseSampler):
    @torch.no_grad()
    def __init__(self, in_channels, Ln1, Ln2, scale=1, eps=0.01, device=None):
        self.in_channels = in_channels
        self.Ln1 = Ln1
        self.Ln2 = Ln2
        self.device = device
        self.scale = scale

        # (H * W, 2), flattened in the same row-major order used by sample().
        meshgrid = get_fixed_coords(self.Ln1, self.Ln2).to(device)
        # (H * W, H * W)
        C = torch.exp(-torch.cdist(meshgrid, meshgrid) / (2 * scale**2))
        # Need to add some regularisation or else the sqrt won't exist
        I = torch.eye(C.size(-1)).to(device)

        # Not memory efficient
        # C = C + (eps**2) * I
        I.mul_(eps**2)  # inplace multiply by eps**2
        C.add_(I)  # inplace add by I
        del I  # don't need it anymore

        # TODO can we support f16 in this class to save gpu memory?

        self.L = torch.linalg.cholesky(C)

        del C  # save memory

    @torch.no_grad()
    def sample(self, N):
        # (N, H * W, H * W) x (N, H * W, n_in) -> (N, H * W, n_in)
        # We can do this in one big torch.bmm, but I am concerned about memory
        # so let's just do it iteratively.
        # L_padded = self.L.repeat(N, 1, 1)
        # z_mat = torch.randn((N, self.Ln1*self.Ln2, self.in_channels)).to(self.device)
        # sample = torch.bmm(L_padded, z_mat)
        samples = torch.zeros((N, self.Ln1 * self.Ln2, self.in_channels)).to(self.device)
        for ix in range(N):
            # (H * W, H * W) * (H * W, n_in) -> (H * W, n_in)
            this_z = torch.randn(self.Ln1 * self.Ln2, self.in_channels).to(self.device)
            samples[ix] = torch.matmul(self.L, this_z)

        # reshape into (N, H, W, n_in)
        sample_rshp = samples.reshape(-1, self.Ln1, self.Ln2, self.in_channels)

        # reshape into (N, n_in, H, W)
        sample_rshp = sample_rshp.transpose(-1, -2).transpose(-2, -3)

        return sample_rshp
