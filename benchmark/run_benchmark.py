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

total_time = time.time() - time_eval
hours = total_time // 3600
mins = (total_time - hours * 3600) // 60
secs = total_time - hours * 3600 - mins * 60

print("TOTAL EVALUATION TIME: %dh %dm %5.3fs" % (hours, mins, secs))
