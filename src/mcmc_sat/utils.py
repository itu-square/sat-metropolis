from z3 import Goal, Solver, BitVecRef, unsat
import arviz as az


def sat_checking_samples(g: Goal,
                         solver_samples: list[dict[str, int]],
                         x: list[BitVecRef]) -> bool:
    """Simple function to verify that all that all samples from a set
    satisfy the Z3 (SMT) problem.

    It assumes that variables are named `xi` with `i` \in Nat \cup
    {0}.

    The parameter `g: Goal` is the one created for bit-blasting
    problems. This should somehow be extracted if used for a Z3 SMT
    problem.

    For samples coming from spur, please use the output of the
    function `reverse_bit_blasting`.

    This function returns True if all samples satisfy the problem `g`
    and False, otherwise.
    """
    for sample in solver_samples:
        s = Solver()
        s.add(g.__copy__())
        for key in sample.keys():
            var_index = int(key.split('x')[1])
            s.add(x[var_index] == sample[key])
        if s.check() == unsat:
            return False
    return True


def save_trace(trace: az.InferenceData, output_filename: str) -> None:
    """Helper function to save output trace as a netcdf file. The
    main reason to define this funciton is to recall our naming
    convention for output files. Recall that you should use this format:

        <problem_id>_<method>_<num_vars>_<num_bits>.nc

    where problem_id ϵ {'db_cacm', 'triangle', roads', 'books',
    'haplotypes'}, method ϵ {'SAT', 'SMT', 'standard_MH',
    'Markov_Basis, 'EM'}, num_vars ϵ ℕ is the number of variables in
    the problem and num_bits ϵ ℕ is the number of for each variable
    (this field is specified if applicable, i.e., if using a SAT
    sampler)
    """
    az.to_netcdf(data=trace, filename=output_filename)


def load_trace(trace_filepath: str) -> az.InferenceData:
    """Simple helper function to load a previously saved trace. See
    documentation of the `save_trace` for the naming convention of
    saved files.
    """
    return az.from_netcdf(trace_filepath)
