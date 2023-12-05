import random
import numpy as np
import arviz as az
from src.mcmc_sat import smt, sat


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
    # samples = get_samples_smt_problem(z3_problem=z3_problem) if backend == 'megasampler' else print('Not implemented yet')
    samples = []
    if backend == 'megasampler':
        samples = smt.get_samples_smt_problem(z3_problem=z3_problem)
    elif backend == 'spur':
        samples = sat.get_samples_sat_problem(z3_problem=z3_problem,
                                              num_vars=num_vars,
                                              num_bits=num_bits,
                                              num_samples=num_samples)

    # run MCMC using the "megasamples" :)
    trace = sample_mh_trace(num_samples, num_chains, samples)

    return trace


# TODO: add function for computing metropolis ratio
def sample_mh_trace(num_samples: int,
                    num_chains: int,
                    solver_samples: [dict[str, int]],
                    uniform_samples: bool = True,
                    var_names: [str] = []) -> az.InferenceData:
    # NOTE: For now, we use the same solver samples for all chains. We
    # could consider using a fresh set of samples for each chain. If
    # we do so, `solver_samples` could be a list of lists of samples
    # and the number of chains could be determine by the length of
    # this list.

    # TODO: Take as input a function with a density function proprotional to the target distribution

    # TODO: Discuss the snippet below in next meeting In a nutshell,
    #       this function can be used in two modes.
    #
    #       First mode, `solver_samples` contains a set of uniform
    #       samples from the SMT problem (e.g., what spur produces).
    #
    #       Second mode, `solver_samples` contains a set of unique
    #       samples to the SMT problem.

    num_solver_samples = 0
    if uniform_samples:
        if num_samples > len(solver_samples):
            raise RuntimeError(
                """`num_samples` must be less than or equal
                to the length of the list of samples in
                `solver_samples`"""
            )
        else:
            num_solver_samples = num_samples
    else:
        num_solver_samples = len(solver_samples)
    # up to here the new snippet

    if var_names == []:
        var_names = solver_samples[0].keys()
    trace = {var: np.ndarray(shape=(num_chains, num_samples), dtype=int)
             for var in var_names}
    for chain in range(num_chains):
        for var in var_names:
            selected_samples = []
            for i in range(num_samples):  # this loop should go before var_names
                # NOTE: the line below is redundant
                # NOTE II: Not really redundant if the solver only produce distinct solutions
                r = random.randint(0, num_solver_samples-1)
                # NOTE II: Might be useful for differential sampling, or using them as starting points
                # NOTE III: We can do weighted sampling to incorporate a prior

                # for now, all samples are accepted

                # a bit inelegant mutable state solution, but we might
                # want to check the metropolis acceptance ratio in the
                # future to decide whether the sample is accepted
                selected_samples.append(solver_samples[r][var])
            trace[var][chain] = selected_samples
    return az.convert_to_inference_data(trace)


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
