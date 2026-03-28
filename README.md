# Decomposition-Based CluVRP Matheuristic Solver

This repository contains code for solving clustered vehicle routing problem instances using:

- Simulated Annealing (SA)
- Iterated Local Search (ILS)
- Hybrid Genetic Search (HGS)

Both routing variants are supported:

- **hard**: clusters must be visited consecutively
- **soft**: clusters must be served by one vehicle but can be leaved before serving every customer

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Mischa-Hermans/cluvrp-solver.git
cd cluvrp-solver
pip install -r requirements.txt
```

---

### 2. Run a single instance

```bash
py -m scripts.run_single
```

Before running, set the configuration:

#### Choose method  
In `configs/methods.py`:
```python
SINGLE_METHOD = "sa"   # "sa", "ils", or "hgs"
```

#### Choose routing variant  
In `configs/routing.py`:
```python
ROUTING_VARIANT = "soft"   # "soft" or "hard"
```

#### Choose instance and runtime  
In `configs/run_single.py`:
```python
SINGLE_INSTANCE_NAME = "K"
SINGLE_TIME_LIMIT_SECONDS = 20.0
SINGLE_BASE_SEED = 42
```

Then run again:

```bash
py -m scripts.run_single
```

This will:
- build an initial solution
- improve it using the selected method
- print initial and final cost

#### Results

The `results/` folder already contains:

- benchmark tables
- tuning outputs
- logs
- example plots for some instances and seeds

---

### 3. Run the full benchmark

```bash
py -m scripts.run_benchmark
```

Before running:

#### Choose method  
In `configs/methods.py`:
```python
BENCHMARK_METHOD = "ils"
```

#### Choose routing variant  
In `configs/routing.py`:
```python
ROUTING_VARIANT = "hard"
```

Optional settings in `configs/benchmark.py`:
- time limit
- checkpoint times
- seeds

The benchmark will:
- run all instances
- save results automatically

Outputs are saved as:

```text
results/tables/hard_cluvrp_ils_results_A_to_K.csv
results/logs/hard_benchmark_runs_ils.pkl
```

---

### 4. Generate plots

```bash
py -m scripts.make_plots
```

Plots are saved in:

```text
results/plots/
```

---

## Repository structure

```text
cluvrp/
├── configs/        # configuration files
├── data/           # instance files (A–K)
├── results/        # outputs (tables, logs, plots)
├── scripts/        # runnable scripts
├── src/cluvrp/     # implementation
├── tests/
```

---

## Main configuration files

- `configs/methods.py` → choose SA / ILS / HGS  
- `configs/routing.py` → choose hard vs soft  
- `configs/run_single.py` → single-instance settings  
- `configs/benchmark.py` → benchmark settings  

---

## Notes

- Gurobi is used for exact TSP solving (license may be required for larger instances)

---
