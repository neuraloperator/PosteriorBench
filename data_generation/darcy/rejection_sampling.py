from data_generation.common import rejection_sampling
from data_generation.darcy.pde import OBSERVED, sample

if __name__ == "__main__":
    rejection_sampling(sample, OBSERVED, "darcy", ref_seed_start=1200, sensor_seed_offset=12345,
                       refs_per_job=35, chunk_size=5000, early_exit=False)
