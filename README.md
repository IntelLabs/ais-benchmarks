# Sampling methods benchmark 

A framework, based on python and numpy, for evaluation of sampling methods. The framework consists of common interfaces 
for sampling methods, an evaluation methodology with different metrics and an automated result and plots generation.

It is our hope that this package helps researches to benchmark their sampling algorithms and compare them to state
of the art implementations, if you find this package useful for your research please consider citing it, see citation
info below. 

### Installation and quickstart
The last supported version is 3.10. guppy3 which is used for CPU and Memory metrics is still not ported to 3.11. 
Once this dependency runs on 3.11 ais-benchmarks should be capable to run on 3.11

### Ubuntu Linux
```shell
sudo apt install python python-pip python-tk
git clone https://github.com/IntelLabs/ais-benchmarks.git ais-benchmarks
cd ais-benchmarks
pip install cython scikit-build
pip install -r requirements.txt
python benchmark/run_benchmark.py
```
### Windows
You need to have python3 and git in your PATH for Windows powershell to find
the commands

Install python 3.10:
https://www.python.org/downloads/release/python-3109/
```shell
git clone https://github.com/IntelLabs/ais-benchmarks.git ais-benchmarks
cd ais-benchmarks
pip install -r requirements.txt
python benchmark/run_benchmark.py
```

### Batching:
All methods assume the first dimension to be the batch dimension to support vectorized implementations. The results
are also in batch form even if there is only one component in the batch dimension.


### Baselines
Several well-known and state-of-the-art algorithms are provided as a baseline for evaluation of new implementation of 
methods. The repo welcomes new algorithms to be added as they become available.

- Deterministic Mixture Adaptive Importance Sampling (DM-AIS) [1]
- Mixture Particle Monte Carlo (M-PMC) [2]
- Layered Adaptive Importance Sampling (LAIS) [3]
- MCMC Metropolis-Hastings [4]
- Nested sampling [5]
- Multi-nested sampling [6]
- Rejection Importance Sampling [7]
- Tree-pyramidal Adaptive Importance Sampling (TP-AIS) [9]
- TODO: Adaptive Population Importance Sampling [8]
- Hierarchical Adaptive Importance Sampling via Exploration and Exploitation. (HiDaisee) [10]


### Evaluation methodology
The benchmark aims to evaluate multiple aspects of sampling algorithms.
- **Sampling efficiency**: How good are the estimates with the minimum number of sample operations, including rejected 
samples.
- **Acceptance rate**: Algorithms that reject samples might exhibit high sampling efficiency and provide uncorrelated
 samples. However, they achieve it at the cost of rejecting samples. This metric considers the number of rejected 
 samples.
- **Proposal quality**: Measures similarity of the proposal to the underlying target distribution.
- **Samples quality**: Measures the similarity of a KDE approximation using the generated samples to the target 
distribution.
- **Time per sample**: Measures the time it required to generate a sample. This metric is useful to quantify the 
overhead that algorithms with more sophisticated proposals and adaptation algorithms have. 
- **Target distribution specific**: Obtain metrics across multiple types of target distributions with different shape
parameters and modality. Combined results provide a good idea of how the methods perform in general while target 
distributions results are also available for analysis.


### Evaluation metrics
As part of the evaluation process, several metrics are computed and stored in the results file.
- Kullback–Leibler divergence (KLD)
- Sampling runtime (T)
- Memory usage (MEM)

Metrics todo:
- Jensen Shannon Divergence (JSD)
- Bhattacharya distance (BD)
- **Normalized Effective Sample Size (NESS)**: Samplers often provide correlated samples, this metric computes the effective
 number of independent samples, by dividing it by the total number of samples generated this metric conveys information
 about how much each sample is worth. [ref]
- Expected value mean squared error (EV-MSE)


### Benchmark configuration and execution
The evaluation process uses a benchmark YAML file that defines the different target distributions that will be 
used to produce the evaluation results. See an example below: 

```yaml
# Define the target densities used for evaluation and label their categories and provide evaluation parameters
targets:
    - normal:
        type: Normal
        params: {loc: 0, scale: 1}
        tags: [low_kurtosis, low_skew, unimodal]
        evaluation:
            batch_size: 2
            nsamples: 1000
            nsamples_eval: 2000

    - gmm:
        type: GMM
        params: {loc: [0, -0.2], scale: [0.01, 0.001], weight:[0.5, 0.5]}
        tags: [high_kurtosis, low_skew, multimodal, close_modes]
        evaluation:
            batch_size: 2
            nsamples: 1000
            nsamples_eval: 2000
``` 

The benchmark YAML file is complemented by a methods YAML file that is used to define the configuration of the methods 
to be evaluated using the benchmark file.
```yaml
# Define the methods to be evaluated and its parameter configuration
methods:
    - name: rejection
      type: CRejectionSampling
      params:
        scaling: 1
        proposal: uniform
        kde_bw: 0.01

    - name: mcmc_mh
      type: CMetropolisHastings    
      params: 
        n_burnin: 10 
        n_steps: 2 
        kde_bw: 0.01 
        proposal_sigma: 0.1
```

Finally the configuration YAML file defines the different metrics that will be used, paths to the results and some
other options for ais-benchmarks to provide other types of results like plots or tables.
```yaml
nreps: 3  # Number of times each experiment is repeated

rseed: 0  # Random seed used for reproducibility

metrics: [KLD, MEM, T]  # Metrics computed

display:
    value: true             # Display 1D and 2D density plots on the screen for debug
    display_path: results/   # If value is true, save the debug plots as a .png in the provided path

    animation: {value: true,                    # Compile the sequence of visualized sampling steps into an animation
                fps: 1,                         # Frames per second used for the animation
                animation_path: results/anim/}  # If value is true, save the plots as a .png in the provided path

output:
    file: results/results.txt
    make_plots: true
    plots_path: results/
    plots_dpi: 1200
```

A benchmark can be executed on the desired methods by using the appropriate script: 
```
run_sampling_bechmark.py benchmark.yaml methods.yaml config.yaml
```

A more thorough example of benchmark and methods file can be found in the provided default benchmarks specified in 
*def_benchmark.yaml*, *def_methods.yaml* and *def_config.yaml*.


## Framework extension
### Implementing new sampling algorithms
Implement the new method by deriving from the sampling_methods.base class that is most suited for your case: 
CMixtureSamplingMethod for general sampling methods and CMixtureISSamplingMethod for IS methods.

TODO: Explain here how a sampling method is implemented, the methods that it must implement and how samples and
weights must be stored and updated. Comment also on the behavior of the importance_sample method and how it
should be used when the sampling method is not IS but simulation like MCMC or Nested.

Reminder to mention the batching and provide an example.

Comment on the optional requirement of methods to handle different dimensionality RV

### Adding target distributions

### Adding metrics



## Authors
- Javier Felip Leon: javier.felip.leon@intel.com

## Citation
Felip et. al. Benchmarking sampling algorithms

## References
[1] Víctor Elvira, Luca Martino, David Luengo, and Mónica F. Bugallo.  Improving populationMonte Carlo: Alternative 
weighting and resampling schemes.Signal Processing, 131(Mc):77–1391, 2017.

[2] Olivier Cappé, Randal Douc, Arnaud Guillin, Jean Michel Marin, and Christian P. Robert. Adaptive importance 
sampling in general mixture classes.Statistics and Computing, 18(4):447–459, 2008.

[3] L Martino,  V Elvira,  D Luengo,  and J Corander. Layered adaptive importance sampling.Statistics and Computing, 
27(3):599–623, may 2017.

[4] W K Hastings.  Monte Carlo sampling methods using Markov chains and their applications.Biometrika, 
57(1):97–109, 1970.

[5] J Skilling. Nested Sampling for Bayesian Computations.Bayesian Analysis, (4):833–860, 2006.

[6] F. Feroz, M.P. Hobson, E. Cameron, and Pettitt A.N.  Importance Nested Sampling and theMULTINEST Algorithm.
Arxiv astro physics, 2014

[7] George Casella, Christian P. Robert, and Martin T. Wells.Generalized Accept-Reject samplingschemes, volume 
Volume 45 ofLecture Notes–Monograph Series, pages 342–347. Institute of Mathematical Statistics, Beachwood, Ohio, 
USA, 2004.

[8] Luca Martino, Victor Elvira, David Luengo, and Jukka Corander.  An adaptive populationimportance sampler: 
Learning from uncertainty.IEEE Transactions on Signal Processing,63(16):4422–4437, 2015

[9] Felip et. al. Tree pyramid adaptive importance sampling

[10] Lu, Xiaoyu, Tom Rainforth, Yuan Zhou, Jan-Willem van de Meent, and Yee Whye Teh. “On Exploration, 
Exploitation and Learning in Adaptive Importance Sampling.” ArXiv:1810.13296 [Cs, Stat], October 31, 2018. 
