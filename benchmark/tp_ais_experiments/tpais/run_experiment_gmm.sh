###
## THIS SCRIPT SPAWNS 16 EVALUATION PROCESSES
###

# Evaluate all TP-AIS with 1,2,3 Dimension gmm benchmark
python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_123D.yaml \
tp_ais_experiments/tpais/def_methods_all.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_123D.txt  2>tp_ais_experiments/tpais/results_gmm_123D_err.txt &

# Evaluate TP-AIS with ESS 60% and 4,5,6 Dimension gmm benchmark
python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_4D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS60.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS60_4D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS60_4D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_5D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS60.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS60_5D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS60_5D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_6D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS60.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS60_6D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS60_6D_err.txt &

# Evaluate TP-AIS with ESS 70% and 4,5,6 Dimension gmm benchmark
python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_4D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS70.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS70_4D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS70_4D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_5D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS70.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS70_5D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS70_5D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_6D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS70.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS70_6D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS70_6D_err.txt &

# Evaluate TP-AIS with ESS 80% and 4,5,6 Dimension gmm benchmark
python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_4D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS80.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS80_4D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS80_4D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_5D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS80.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS80_5D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS80_5D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_6D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS80.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS80_6D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS80_6D_err.txt &

# Evaluate TP-AIS with ESS 90% and 4,5,6 Dimension gmm benchmark
python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_4D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS90.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS90_4D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS90_4D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_5D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS90.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS90_5D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS90_5D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_6D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS90.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS90_6D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS90_6D_err.txt &

# Evaluate TP-AIS with ESS 95% and 4,5,6 Dimension gmm benchmark
python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_4D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS95.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS95_4D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS95_4D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_5D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS95.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS95_5D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS95_5D_err.txt &

python3 run_benchmark.py tp_ais_experiments/benchmarks/def_benchmark_gmm_6D.yaml \
tp_ais_experiments/tpais/def_methods_TPAIS95.yaml tp_ais_experiments/tpais/def_config.yaml \
tp_ais_experiments/tpais/results_gmm_TPAIS95_6D.txt  2>tp_ais_experiments/tpais/results_gmm_TPAIS95_6D_err.txt &
