import json
import numpy as np

class GammaProcess:
    """Gamma arrival process."""
    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.

        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate # time interval = mean, so rate = 1 / mean
        self.cv_ = cv # cv = sigma / mean = 1 / sqrt(k)
        self.shape = 1 / (cv * cv) # shape = k
        self.scale = cv * cv / arrival_rate # scale = theta

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_
    
    def generate_arrival_intervals_yhc(self, arrival_rate: float, cv: float, request_count: int, seed: int = 0):
        np.random.seed(seed)
        shape = 1 / (cv * cv)
        scale = cv * cv / arrival_rate
        intervals = np.random.gamma(shape, scale, size=request_count).tolist()
        return intervals

seed = 0
request_count = 500

# generate random step
np.random.seed(seed)
random_step_list = np.random.randint(30, 51, request_count)

f = open("H100_SD_FP16_img512_trace.json", "w")
trace = {
    "random_step_list": random_step_list.tolist()
}

process = GammaProcess(1.0, 1.0)

# "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45"
arrival_rate_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] # req/s
cv_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] # degree of suddenness
for arrival_rate in arrival_rate_list:
    for cv in cv_list:
        trace[f"rate={arrival_rate},cv={cv}"] = process.generate_arrival_intervals_yhc(arrival_rate, cv, request_count, 0)

json.dump(trace, f)