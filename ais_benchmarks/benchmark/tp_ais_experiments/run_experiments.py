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
from utils.misc import time_to_hms
def_path = str(pathlib.Path(__file__).parent.absolute()) + os.sep


n_cpus = os.cpu_count()
n_jobs = 2
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

print("CPUs available: ", os.cpu_count())
n_jobs = min(n_jobs, n_cpus)
print("Added %d jobs with %d methods and %d benchmarks" % (len(jobs_cmd), len(method_files), len(benchmark_files)))
active_jobs = [None] * n_jobs
active_jobs_flag = [False] * n_jobs
active_jobs_cmd = [""] * n_jobs
stream_readers = [False] * n_jobs
finished_jobs = list()
init = True

t_ini = time.time()
while any(active_jobs_flag) or init:
    init = False
    for num in range(n_jobs):
        if not active_jobs_flag[num]:
            if len(jobs_cmd)>0:
                job_cmd = jobs_cmd.pop()
                print("JOB #%03d START: %s" % (num, job_cmd))
                active_jobs[num] = subprocess.Popen(job_cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                active_jobs_flag[num] = True
                stream_readers[num] = CNonBlockingStreamReader(active_jobs[num].stdout)
                active_jobs_cmd[num] = job_cmd
                # Set each process to its own CPU
                os.sched_setaffinity(active_jobs[num].pid, {num})

    for num, job in enumerate(active_jobs):
        if active_jobs_flag[num] and job.poll() is not None:
            print("JOB #%03d FINISHED: %d | %s" % (num, job.poll(), str(job)))
            active_jobs_flag[num] = False
            finished_jobs.append(active_jobs_cmd[num])

        if active_jobs_flag[num]:
            out = stream_readers[num].read_last_and_clear()
            if out != "":
                print("JOB #%03d: %s" % (num, out))

    # Obtain experiment execution runtime
    hr, mn, sec = time_to_hms(time.time() - t_ini)

    # Count running jobs
    njobs_running = 0
    for job_running in active_jobs_flag:
        if job_running:
            njobs_running += 1

    print("Running %d jobs. Queued: %d. Done: %d. Elapsed: %02d:%02d:%5.3f" %
          (njobs_running, len(jobs_cmd), len(finished_jobs), hr, mn, sec))

    time.sleep(1)

print("JOB RUNNING COMPLETE. List:")
for cmd in finished_jobs:
    print(" - ", cmd)

hr, mn, sec = time_to_hms(time.time() - t_ini)
print("TOOK: %d:%d:%5.3f." % (hr, mn, sec))
