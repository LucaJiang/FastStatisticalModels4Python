#!/usr/bin/env bash
set -e

echo "Running K-Means Benchmark Sweep..."
python3 experiments/kmeans/bench_kmeans.py --n-samples 10000 --impl numpy numba loops jax --output-csv experiments/results/kmeans_n10k.csv
python3 experiments/kmeans/bench_kmeans.py --n-samples 50000 --impl numpy numba loops jax --output-csv experiments/results/kmeans_n50k.csv
python3 experiments/kmeans/bench_kmeans.py --n-samples 200000 --impl numpy numba jax --output-csv experiments/results/kmeans_n200k.csv
python3 experiments/kmeans/bench_kmeans.py --n-samples 500000 --impl numpy numba jax --output-csv experiments/results/kmeans_n500k.csv

echo "Running Permutation Test Benchmark Sweep..."
python3 experiments/permutation_test/bench_permtest.py --n1 1000 --n2 1000 --r 2000 --impl numpy multiprocessing threads numba jax --output-csv experiments/results/perm_n2k.csv
python3 experiments/permutation_test/bench_permtest.py --n1 5000 --n2 5000 --r 2000 --impl numpy multiprocessing threads numba jax --output-csv experiments/results/perm_n10k.csv
python3 experiments/permutation_test/bench_permtest.py --n1 10000 --n2 10000 --r 2000 --impl numpy multiprocessing threads numba jax --output-csv experiments/results/perm_n20k.csv

echo "Done running sweeps."
