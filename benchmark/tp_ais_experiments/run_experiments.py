"""
For each method in methods subdir
    For each benchmark in benchmarks subdir
        Create a job description


While job descriptions is not empty
    wait for a processing slot
        Dispatch a job
        Remove the job from the pending job list

Combine results
"""

import os
import pathlib
import subprocess
import time
from utils.misc import CNonBlockingStreamReader
def_path = str(pathlib.Path(__file__).parent.absolute()) + os.sep


n_jobs = 4
interpreter = "python3.7"
run_bench = def_path + ".." + os.sep + "run_benchmark.py"
benchmarks_subdir = def_path + "benchmarks" + os.sep
methods_subdir = def_path + "methods" + os.sep
results_subdir = def_path + "results" + os.sep
c_file = def_path + "def_config.yaml"


jobs_cmd = list()
benchmark_files = os.listdir(benchmarks_subdir)
method_files = os.listdir(methods_subdir)
for m_file in method_files:
    for b_file in benchmark_files:
        o_file = results_subdir + m_file + b_file + ".dat"
        cmd = "%s %s %s %s %s %s" % \
              (interpreter, run_bench, benchmarks_subdir + b_file, methods_subdir + m_file, c_file, o_file)
        jobs_cmd.append(cmd)
        print(cmd)

print("Added %d jobs with %d methods and %d benchmarks" % (len(jobs_cmd), len(method_files), len(benchmark_files)))
active_jobs = [None] * n_jobs
active_jobs_flag = [False] * n_jobs
stream_readers = [False] * n_jobs
while len(jobs_cmd) > 0:
    for num in range(n_jobs):
        if not active_jobs_flag[num]:
            job_cmd = jobs_cmd.pop()
            print("JOB #%d START: %s" % (num, job_cmd))
            active_jobs[num] = subprocess.Popen(job_cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            active_jobs_flag[num] = True
            stream_readers[num] = CNonBlockingStreamReader(active_jobs[num].stdout)

    for num, job in enumerate(active_jobs):
        if active_jobs_flag[num] and job.poll() is not None:
            print("JOB #%d FINISHED: %d | %s" % (num, job.poll(), str(job)))
            active_jobs_flag[num] = False

        if active_jobs_flag[num]:
            out = stream_readers[num].read_last_and_clear()
            if out != "":
                print("JOB #%d: %s" % (num, out))

    print("Running %d jobs. Queued: %d" % (len(active_jobs), len(jobs_cmd)))
    time.sleep(1)
