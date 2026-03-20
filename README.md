CluVRP – Heuristic and Metaheuristic Solver

This repository contains code to construct and improve solutions for CluVRP instances using routing heuristics, local search, and simulated annealing.

--------------------------------------------------
Getting started
--------------------------------------------------

Clone the repository and move into the project folder:

git clone <repo-link>
cd cluvrp

Install the required packages:

pip install -r requirements.txt

--------------------------------------------------
Data
--------------------------------------------------

The instance files are available in:

data/instances-set1/
data/instances-set2/

Each folder contains files like:
A.gvrp, B.gvrp, C.gvrp, ...

If instance data is stored somewhere else, you can change the paths in:
configs/default.py

--------------------------------------------------
Run a single instance
--------------------------------------------------

From the project root, run:

py -m scripts.run_single

This will:
- load one instance (default is set in the script)
- construct an initial solution
- improve it using simulated annealing
- print the final solution cost

To change the instance, edit:
scripts/run_single.py

--------------------------------------------------
Run the full benchmark
--------------------------------------------------

To run all instances and generate results:

py -m scripts.run_benchmark

Results will be written to:

results/tables/

--------------------------------------------------
Plots
--------------------------------------------------

To generate plots (routes or convergence):

py -m scripts.make_plots

Plots will be saved in:

results/plots/

--------------------------------------------------
Output
--------------------------------------------------

During execution, the program prints:
- initial solution cost
- improvements during search
- final solution cost

Benchmark runs additionally produce CSV tables with results.