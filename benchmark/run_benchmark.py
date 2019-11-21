import time
import sys
from benchmark.CBenchmark import CBenchmark


benchmark = CBenchmark()

# TODO: make this paths relative to this file
b_file = "def_benchmark.yaml"
m_file = "def_methods.yaml"

if len(sys.argv) == 3:
    b_file = sys.argv[1]
    m_file = sys.argv[2]

time_eval = time.time()

benchmark.run(benchmark_file=b_file, methods_file=m_file)

print("TOTAL EVALUATION TIME: %f hours" % ((time.time() - time_eval) / 3600.0))
