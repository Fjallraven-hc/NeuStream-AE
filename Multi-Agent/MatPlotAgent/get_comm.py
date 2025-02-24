import re

def clean_log(log_file: str):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        req_data = re.findall(r'Request id: (\d+), e2e: ([\d.]+), real_time: ([\d.]+)', line)
        if not req_data:
            continue
        else:
            req_data = req_data[0]
        e2e = float(req_data[1])
        real = float(req_data[2])
        data.append(1-real/e2e)
    return data

data = clean_log('./log/final.log')
avg_rate = sum(data)/len(data)

fn = lambda x: round(x*100, 5)
print(f"average_rate: {fn(avg_rate)}")
