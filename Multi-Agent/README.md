# Prerequisites

## Environment Setup

Follow these steps to set up the environment and install dependencies:

```bash
# Create and activate a Conda environment with Python 3.10
conda create -n <venv_name> python=3.10 -y  
conda activate <venv_name>

# Navigate to the vLLM directory and install dependencies
cd vllm_0_6/vllm  
pip install -r requirements-cuda.txt  
pip install -e .  

# Navigate to the MatPlotAgent directory and install additional dependencies
cd ../../MatPlotAgent  
pip install -r requirements.txt  
```

### Running on an Existing Machine

If the environment is already set up on this machine, simply activate the virtual environment:

```bash
conda activate vllm_0_6
```

### Important Note

Before running new experiments, check whether the `log` directory contains important data. If needed, rename the directory to preserve previous results.

---

# Reproducing Figures 17 and 19

## Running End-to-End Experiments

Execute the following command to run the end-to-end experiments:

```bash
bash ./run_end2end.sh  
```

## Parsing Logs and Generating Figures

### Generate Figure 17

```bash
python plot_end2end.py  
```

### Generate Figure 19

```bash
python plot_norm.py  
```

---

# Reproducing Table 2

## Communication Cost

To compute the communication cost in Table 2, run:

```bash
bash ./run_comm.sh
python get_comm.py
```

## Scheduling Cost

To compute the scheduling cost, execute:

```bash
bash ./run_sched.sh
python get_sched.py
```

## Memory and Switch Cost

To compute the memory and switch costs:

```bash
# Run tests for different model types
python test_switch.py --type codellama
python test_switch.py --type llava
```

This will output the switch cost. To check the memory overhead, search for "memory overhead" in the output.

