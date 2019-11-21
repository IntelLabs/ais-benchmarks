import time
import sys
from benchmark.CBenchmark import CBenchmark


benchmark = CBenchmark()

if len(sys.argv) == 3:
    benchmark.load(sys.argv[1], sys.argv[2])
else:
    benchmark.load("def_benchmark.yaml", "def_methods.yaml")

time_eval = time.time()

benchmark.run()

print("TOTAL EVALUATION TIME: %f hours" % ((time.time() - time_eval) / 3600.0))
