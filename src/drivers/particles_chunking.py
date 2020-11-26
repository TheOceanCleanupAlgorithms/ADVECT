import pandas as pd

def chunk_particles(num_kernels: int, p0: pd.DataFrame):
    """
    Given a number of kernels to run in parallel with different chunks of particles,
    compute chunks of almost same size, with +1/-1 in number of parts to fit all particles.
    """
    num_particles = len(p0)

    min_num_particles_per_kernel = num_particles // num_kernels
    num_kernels_with_additional_part = num_particles % num_kernels

    chunks = []
    current_idx = 0

    for idx in range(num_kernels_with_additional_part):
        next_idx = current_idx + min_num_particles_per_kernel + 1
        chunks.append(p0.loc[(current_idx <= p0.p_id) & (p0.p_id < next_idx)])
        current_idx = next_idx

    for idx in range(num_kernels - num_kernels_with_additional_part):
        next_idx = current_idx + min_num_particles_per_kernel
        chunks.append(p0.loc[(current_idx <= p0.p_id) & (p0.p_id < next_idx)])
        current_idx = next_idx

    return chunks