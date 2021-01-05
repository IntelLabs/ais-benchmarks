import time
import sys
import pathlib
from os import sep
from benchmark.CBenchmark import CBenchmark
from utils.misc import time_to_hms


benchmark = CBenchmark()

# Make the default paths relative to this file
def_path = str(pathlib.Path(__file__).parent.absolute()) + sep

b_file = def_path + "def_benchmark.yaml"
m_file = def_path + "def_methods.yaml"
c_file = def_path + "def_config.yaml"

out_file = def_path + "results.txt"

if len(sys.argv) == 5:
    b_file = sys.argv[1]
    m_file = sys.argv[2]
    c_file = sys.argv[3]
    out_file = sys.argv[4]

time_eval = time.time()

benchmark.run(benchmark_file=b_file, methods_file=m_file, config_file=c_file, out_file=out_file)

total_time = time.time() - time_eval
h, m, s = time_to_hms(total_time)

print("TOTAL EVALUATION TIME: %dh %dm %5.3fs" % (h, m, s))
