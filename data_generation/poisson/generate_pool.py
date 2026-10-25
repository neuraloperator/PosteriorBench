from data_generation.common import generate_pool
from data_generation.poisson.pde import sample

if __name__ == "__main__":
    generate_pool(sample, "poisson", base_seed=777)
