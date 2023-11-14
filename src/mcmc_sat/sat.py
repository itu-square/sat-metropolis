from subprocess import call
from z3 import Goal, BitVecSort, Bool, And, BVAddNoOverflow, BVMulNoOverflow, BVSubNoUnderflow, Then, simplify, is_app_of, Z3_OP_NOT, Not, BoolRef
import math
import random
import numpy as np
import re


# NOTE: The function below creates a new variable for each bit in the
#       bit-vector.  Then, it maps the correspoding variable of the
#       bit vectors in `var_list` to the corresponding bit created
#       below.  The process is a bit convoluted. First, it creates the
#       Bool variable and adds it to a bitmap. Then, it creates a mask
#       that will be used to isolate the bit in the bit-vector
#       above. This last step is done by adding the final
#       constraint. The mask is of binary number of the kind 2^i with
#       i the bit position we are defining. Note that the binary
#       representation of 2^i numbers contains only a single bit equal
#       1. Then the operation and (&) is used with the bit-vector in
#       var_list to isolate the bit of interest. The result of this
#       operation is compared with the mask again, this is just to
#       create predicate that return 1 if the bit is 1 and 0 if it is
#       0. Finally, the result of this operation must be equal to the
#       bit representating that position of the bit-vector. This is
#       added as a constraint to the problem.

def add_bool_vars_to_goal(g: Goal, var_list: [BitVecSort]):
    """This function creates a Bool variable for each bit in the
    BitVectors in `var_list`.

    Note that the name of the variables is fixed `x_ji` where j is the
    index of the BitVector in `var_list` and i is the index of bit in
    the corresponding BitVector.
    """
    # NOTE: We could try to obtain the variable name from `var_list`
    bitmap = {}
    num_vars = len(var_list)
    for j in range(num_vars):
        num_bits = var_list[j].size()  # returns #bits in `var_list[j]`
        for i in range(num_bits):
            bitmap[var_list[j], i] = Bool(f'x{j}{i}')
            mask = BitVecSort(num_bits).cast(math.pow(2, i))
            g.add(bitmap[(var_list[j], i)] == ((var_list[j] & mask) == mask))


# NOTE: The code in the function below is taken from Maja's
# implementation as it was. I wonder whether we do not need to add for
# each sub-sum?

# adds a constraint regarding the summation of all elements in xs
# (removes the overflow) and returns the variable
def addition_does_not_overflow(xs: [], signed=False):
    sofar      = 0
    noOverflow = True
    for x in xs:
        noOverflow = And(noOverflow, BVAddNoOverflow(x, sofar, signed))
        sofar += x
    return noOverflow


# same as above but multiplication
def multi_does_not_overflow(xs, signed=False):
    sofar      = 1
    noOverflow = True
    for x in xs:
        noOverflow = And(noOverflow, BVMulNoOverflow(x, sofar, signed))
        sofar *= x
    return noOverflow


# same as above
# most likely substraction from first array element to the last
def sub_does_not_underflow(xs, signed=False):
    sofar      = 0
    noUnderflow = True
    for x in xs:
        noUnderflow = And(noUnderflow, BVSubNoUnderflow(x, sofar, signed))
        sofar = x-sofar
    return noUnderflow


def convert_to_cnf_and_dimacs(g: Goal):
    # copied from De Moura's post -> https://stackoverflow.com/a/13059908
    t = Then('simplify', 'bit-blast', 'tseitin-cnf')
    subgoal = t(g)
    assert len(subgoal) == 1

    # Maja's conversion code (seems to be correct, although can be
    # refactored, and double-checked)
    var_count = 1
    constraint_count = 0
    varibles_number = {}
    varibles_var = {}
    negation = False
    dimacs_format = [[]]

    for c in subgoal[0]:
        constraint_count += 1
        dimacs_format.append([])
        for i in range(c.num_args()):
            # Save varible names in dictonary
            # print(c.arg(i))
            if (c.num_args() == 1):
                if (is_app_of(c, Z3_OP_NOT)):
                    var = simplify(Not(c.arg(i)))
                    negation = True
            if (is_app_of(c.arg(i), Z3_OP_NOT)):
                var = simplify(Not(c.arg(i)))
                negation = True
            else:
                var = c.arg(i)
            if (var not in varibles_var):
                varibles_var[var] = var_count
                varibles_number[var_count] = var
                var_count += 1
            if (negation):
                dimacs_format[constraint_count].append(-varibles_var[var])
            else:
                dimacs_format[constraint_count].append(varibles_var[var])
            negation = False
        if (c.num_args() == 0):
            var = c
            if (var not in varibles_var):
                varibles_var[var] = var_count
                varibles_number[var_count] = var
                var_count += 1
            dimacs_format[constraint_count].append(varibles_var[var])
        dimacs_format[constraint_count].append(0)

    # appending heading line
    n_varibles = len(varibles_var)
    n_constraints = len(dimacs_format)-1
    s = "p cnf " + str(n_varibles) + " " + str(n_constraints)
    dimacs_format[0].append(s)
    return (dimacs_format, n_varibles, varibles_number)


def save_dimacs(g: Goal, output_filepath: str) -> (int, dict):
    # NOTE: We return n_variables because it is later used to parse
    #       the output of spur.
    #       Also, we return the map variables_number because we need
    #       to map back the results from spur to its Z3 variables.
    (dimacs_format, n_varibles, varibles_number) = convert_to_cnf_and_dimacs(g)
    with open(output_filepath, 'w') as file:
        for row in dimacs_format:
            file.write(' '.join([str(item) for item in row]))
            file.write('\n')
    return (n_varibles, varibles_number)


def execute_spur(input_filepath: str, num_samples: int = 10000) -> None:

    """Executes spur on the specified input file `input_filepath`. By
    default, it generates 10000 samples. The samples are created on
    the same directory as the input file. The name of the output file
    is `samples_<name_of_input_file>.txt`.

    The function assumes that the spur executable is accessible
    by calling `spur`.
    """
    call(['spur',                  # - spur command (hardcoded, it
                                   #   assumes accessible for this user)
          '-s', str(num_samples),  # - number of samples
          '-cnf',                  # - Input format (DIMACS cnf)
          input_filepath])         # - input file path


def __repl_fun(match):
    # randint, discrete uniform distribution
    return str(random.randint(0, 1))


def parse_spur_samples(input_dir: str,
                       input_file: str,
                       num_samples: int,
                       num_variables: int) -> list[list[bool]]:

    spur_samples_filepath = f'{input_dir}/samples_{input_file[:-4]}.txt'
    n = num_samples
    m = num_variables
    samples1 = np.zeros((n, m), dtype=np.int8)
    # counter = 0
    with open(spur_samples_filepath, 'r') as f:
        i = 0
        for line in f:
            if ((line[0]).isdigit()):
                n_ = int(line.split(',')[0])
                sampel2 = line.split(',')[1]
                sampel = sampel2.split('\n')[0]
                for j in range(n_):
                    replaced = re.sub('\*', __repl_fun, sampel)
                    samples1[i] = list(map(int, replaced))
                    i += 1
    return samples1


def map_spur_samples_to_z3_vars(map_number_z3_var: dict[int, BoolRef],
                                num_variables: int,
                                spur_parsed_samples: list[list[bool]]
                                ) -> dict[str, list[bool]]:
    """This function takes as in put the mapping of integer to Z3 Bool
    variables returned by `convert_to_cnf_and_dimacs` in variable
    `variables_number`, and converts it into a dictonary with variable
    names as strings and the list of (spur) sampled values for each
    variable.

    For now, it is also required to specify, the number of variables
    in `map_number_z3_var`, althought this information could be
    obtained from `map_number_z3_var`.

    In this funciton, we still work with blasted variables.
    """
    # map_number_z3_var = variables_number
    variable_values = {}
    for i in range(num_variables):
        z3_var_str = str(map_number_z3_var[i+1])
        variable_values[z3_var_str] = spur_parsed_samples[:, i]
    return variable_values


def reverse_bit_blasting(variable_values: dict[str, list[bool]],
                         num_samples: int,
                         num_vars: int,
                         num_bits: int) -> list[dict[str, int]]:
    map_var_name_samples = {}
    for i in range(num_vars):
        var_name = f'x{i}'
        map_var_name_samples[var_name] = np.zeros(num_samples, dtype=np.int8)
        for j in range(num_bits):
            bit_j_value = 2**j * variable_values[f'{var_name}{j}']
            map_var_name_samples[var_name] += bit_j_value

    # transform into format for the mcmc Metropolis-Hastings function
    # NOTE: this step could be avoided if we modify slightly the
    # mcmc.sample_mh_trace fucntion. In fact, mcmc.sample_mh_trace
    # maps back the dictionary into the shape in
    # `map_var_name_samples`
    var_names = list(map_var_name_samples.keys())
    # num_samples = len(map_var_name_samples[var_names[0]])
    solver_samples = []
    for i in range(num_samples):
        sample = {v: map_var_name_samples[v][i] for v in var_names}
        solver_samples.append(sample)
    return solver_samples
