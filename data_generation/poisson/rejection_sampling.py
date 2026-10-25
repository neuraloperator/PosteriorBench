from data_generation.common import rejection_sampling
from data_generation.poisson.pde import OBSERVED, sample

if __name__ == "__main__":
    rejection_sampling(sample, OBSERVED, "poisson", ref_seed_start=5800, sensor_seed_offset=999,
                       refs_per_job=10, chunk_size=10000, early_exit=True)
