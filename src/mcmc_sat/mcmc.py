import random
import numpy as np
import arviz as az
from src.mcmc_sat import smt, sat
from z3 import Solver, Int, Sum
from typing import Callable


def sample_mh_trace_from_z3_model(backend: str,
                                  z3_problem,
                                  num_vars: int = None, # mandatory for spur
                                  num_bits: int = None, # mandatory for spur
                                  num_samples: int = 10000,
                                  num_chains: int = 4):
    """
    TODO: Document
    TODO: We must implement different behaviour depending on whether
          we use megasampler or spur.  The backend should be specified
          in the paramter `backend` ϵ {'megasampler', 'spur'} If we
          must use the appropriate format for the z3_problem.  That
          is, an object of type Solver for megasampler and an object
          of type goal for spur.
    """

    samples = []
    if backend == 'megasampler':
        samples = smt.get_samples_smt_problem(z3_problem=z3_problem,
                                              num_samples=num_samples)
    elif backend == 'spur':
        samples = sat.get_samples_sat_problem(z3_problem=z3_problem,
                                              num_vars=num_vars,
                                              num_bits=num_bits,
                                              num_samples=num_samples)

    # run MCMC using the samples from spur or megasampler
    trace = sample_mh_trace(num_samples, num_chains, samples)

    return trace


# TODO: add function for computing metropolis ratio
def sample_mh_trace(num_samples: int,
                    num_chains: int,
                    solver_samples: [dict[str, int]],
                    f: Callable[[list[int]], float] = lambda x: 1,  # by default all samples have the same probability
                                                                    # this should mean uniform prior (if samples uniformly distributed)
                    var_names: [str] = []) -> az.InferenceData:
    # NOTE: For now, we use the same solver samples for all chains. We
    # could consider using a fresh set of samples for each chain. If
    # we do so, `solver_samples` could be a list of lists of samples
    # and the number of chains could be determine by the length of
    # this list.

    # TODO: Take as input a function with a density function
    # proprotional to the target distribution

    # Informing users if there is mismatch between the specified
    # number of samples to generate and the number of samples in
    # `solver_samples`
    num_solver_samples = len(solver_samples)
    if num_samples > num_solver_samples:
        # TO-DISCUSS: Is this a good solution for this case?
        print(f'The parameter `solver_samples` only contains {num_solver_samples} samples. Thus, every chain will contain {num_solver_samples} instead of {num_samples}. Try running the SAT/SMT sampler longer to obtain more samples.')
        num_samples = num_solver_samples
    elif num_samples < num_solver_samples:
        print(f'The parameter `solver_samples` contains {num_solver_samples}, which is larger that the number of sample per chain specified: {num_samples}. Every chain will contain {num_samples} samples as specified. Nevertheless, we inform you that you can produce chains of up to {num_solver_samples} if you specify so in {num_samples}.')

    # up to here the new snippet

    # obtain variable names from samples if not specified
    if var_names == []:
        var_names = solver_samples[0].keys()
    trace = {var: np.ndarray(shape=(num_chains, num_samples), dtype=int)
             for var in var_names}

    # TODO: make a regular `def` function?
    should_I_stay_or_should_I_go = lambda s_i, s_prime, alpha, u: s_prime if u <= alpha else s_i

    # TODO: make a regular `def` function?
    get_var_trace = lambda var_name, s: [s_i[var_name] for s_i in s]

    # generate each chain separately
    for chain in range(num_chains):
        # init memory for trace
        s: [dict[str, int]] = []
        # random shuffle of samples (to produce different chains with
        # the same solver sample, TO-DISCUSS)
        np.random.shuffle(solver_samples)
        # take inital state from sample
        s.append(solver_samples[0])
        # s_i: dict[str, int] = s_0
        for i in range(num_samples-1):
            # 1. draw a sample for candidate next state
            s_prime: dict[str, int] = solver_samples[i+1]
            # 2. compute acceptance ratio (s[i] is the current state)
            alpha: float = f(s_prime)/f(s[i])
            # 3. draw a sample uniformly from the unit interval -> U(0,1)
            u: float = np.random.uniform(0, 1)
            # 4. decide whether you move to next state
            s_next: dict[str, int] = should_I_stay_or_should_I_go(s[i],
                                                                  s_prime,
                                                                  alpha, u)
            # 5. add it to trace
            s.append(s_next)
        # switch to input format for arviz
        for var in var_names:
            trace[var][chain] = get_var_trace(var, s)
    return az.convert_to_inference_data(trace)


def sample_mh_trace_from_conf_matrix_smt(A: np.ndarray,
                                         y: np.ndarray,
                                         num_samples: int = 10000):
    """TODO: Document Function to sample directly from a problem
    specified using a configuration matrix A, and a set of
    observations y. The function automatically builds a Z3 model and
    samples using megasampler.
    """
    num_vars = A.shape[1]
    num_ys   = y.shape[0]

    x = [Int(f'x{i}') for i in range(num_vars)]

    s = Solver()

    for i in range(num_vars):
        s.add(x[i] >= 0)

    for i in range(len(y)):
        # Maja's original
        # vars_ = []
        # for j in range(num_vars):
        #     if(A[i][j]==1):
        #         vars_.append(x[j])
        vars_ = [x[j] for j in range(num_vars) if A[i][j] == 1]  # alternative
        s.add(Sum(vars_) == y[i])

    trace = sample_mh_trace_from_z3_model(backend='megasampler',
                                          z3_problem=s,
                                          num_samples=num_samples)

    return trace


def __get_sample(var, samples):
    return lambda i: samples[i][var]


def sample_mh_trace_uniform_hardcoded(
        num_samples: int,
        num_chains: int,
        solver_samples: dict[str, int],
        var_names: [str] = []) -> dict[str, [[int]]]:
    if var_names == []:
        # if not specified, we get var names from the sample
        var_names = solver_samples[0].keys()
    trace = {var: [] for var in var_names}
    for var in var_names:
        map_for_samples = np.vectorize(__get_sample(var, solver_samples))
        samples_vector = np.random.randint(0, num_samples-1,
                                           size=(num_chains, num_samples))
        trace[var] = map_for_samples(samples_vector)
    return trace
