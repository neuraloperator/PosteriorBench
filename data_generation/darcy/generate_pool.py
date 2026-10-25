from data_generation.common import generate_pool
from data_generation.darcy.pde import sample

if __name__ == "__main__":
    generate_pool(sample, "darcy", base_seed=888)
