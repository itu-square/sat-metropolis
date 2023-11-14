import random
import numpy as np
import arviz as az


# TODO: add function for computing metropolis ratio
def sample_mh_trace(num_samples: int,
                    num_chains: int,
                    solver_samples: [dict[str, int]],
                    var_names: [str] = []) -> az.InferenceData:
    # NOTE: For now, we use the same solver samples for all chains. We
    # could consider using a fresh set of samples for each chain. If
    # we do so, `solver_samples` could be a list of lists of samples
    # and the number of chains could be determine by the length of
    # this list.

    if num_samples > len(solver_samples):
        raise RuntimeError(
            """`num_samples` must be less than or equal
            to the length of the list of samples in
            `solver_samples`"""
        )
    if var_names == []:
        var_names = solver_samples[0].keys()
    trace = {var: np.ndarray(shape=(num_chains, num_samples), dtype=int)
             for var in var_names}
    for chain in range(num_chains):
        for var in var_names:
            selected_samples = []
            for i in range(num_samples):  # this loop should go before var_names
                r = random.randint(0, num_samples-1)  # NOTE: this is redundant
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
