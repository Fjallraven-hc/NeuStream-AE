import time

from Palette_pipe_for_clockwork import Palette_pipe

device = "cuda:4"
pipe = Palette_pipe(device=device)

batch_request = [-1]

num_steps = 1000
for _ in range(5):
    begin = time.perf_counter()
    pipe(batch_request, num_steps)
    end = time.perf_counter()
    print(f"batch_size={len(batch_request)}, latency = {end - begin}")
