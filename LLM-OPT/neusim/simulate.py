import json
import numpy as np
import os
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--num_dedup',
        required=False,
        type=int,
        default=3,
    )
    return parser.parse_args()

# 计算均方误差（MSE）
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 计算决定系数（R^2）
def calculate_r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

def simu_prefill(filename: str, num_dedup: int):
    with open(filename) as f:
        log = list(f)

    delim = num_dedup+2
    y = []
    sum_len = []
    sum_len2 = []
    for i in range(len(log)//delim):
        seql = int(log[i*delim].split(" ")[-1])
        bs_ls = np.array(json.loads(log[i*delim+1].split(": ")[-1]))
        time_ls:np.ndarray = np.array([np.array(json.loads(log[i*delim+j].split(": ")[-1])) for j in range(2, 2+num_dedup)]).min(axis=0)

        y.extend(time_ls.tolist())
        sum_len.extend((bs_ls*seql).tolist())
        sum_len2.extend((bs_ls*(seql**2)).tolist())

    y = np.array(y)
    X = np.array([sum_len, sum_len2]).T
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # evaluate
    y_pred = X.dot(theta)
    mse = calculate_mse(y, y_pred)
    r_squared = calculate_r_squared(y, y_pred)
    print("均方误差(MSE):", mse)
    print("决定系数(R^2):", r_squared)

    return theta

def linearRegression_2d(x, y):
    A = np.vstack((x, np.ones(len(x)))).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

def simu_decode(filename: str):
    with open(filename) as f:
        logs = list(f)
    y = np.array(json.loads(logs[1]))[1:]
    x = np.array(json.loads(logs[2]))[1:]
    k, b = linearRegression_2d(x, y)

    return k, b


if __name__ == '__main__':
    args = _parse_args()
    files = os.listdir(args.dir)

    
    for f in files:
        if f.startswith('prefill'):
            theta = simu_prefill(f"{args.dir}/{f}", args.num_dedup)
        elif f.startswith('decode'):
            kb = simu_decode(f"{args.dir}/{f}")
    
    output = {
        'theta': theta.tolist(),
        'kb': kb
    }
    with open(f"{args.dir}/output.log", 'w') as file:
        file.write(json.dumps(output))

