from z3 import Goal, Solver, BitVecRef, unsat


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
