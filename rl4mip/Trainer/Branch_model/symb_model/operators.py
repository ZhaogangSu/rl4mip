import torch
from collections import OrderedDict, defaultdict

NODE_FEATURES = ['type_0', 'type_1', 'type_2', 'type_3', 'coef_normalized', 'has_lb', 'has_ub', 'sol_is_at_lb', 'sol_is_at_ub', 'sol_frac', 'basis_status_0', 'basis_status_1', 'basis_status_2', 'basis_status_3', 'reduced_cost', 'age', 'sol_val', 'inc_val', 'avg_inc_val']
KHALI_FEATURES = ['acons_max1', 'acons_max2', 'acons_max3', 'acons_max4', 'acons_mean1', 'acons_mean2', 'acons_mean3', 'acons_mean4', 'acons_min1', 'acons_min2', 'acons_min3', 'acons_min4', 'acons_nb1', 'acons_nb2', 'acons_nb3', 'acons_nb4', 'acons_sum1', 'acons_sum2', 'acons_sum3', 'acons_sum4', 'acons_var1', 'acons_var2', 'acons_var3', 'acons_var4', 'cdeg_max', 'cdeg_max_ratio', 'cdeg_mean', 'cdeg_mean_ratio', 'cdeg_min', 'cdeg_min_ratio', 'cdeg_var', 'coefs', 'coefs_neg', 'coefs_pos', 'frac_down_infeas', 'frac_up_infeas', 'nb_down_infeas', 'nb_up_infeas', 'nnzrs', 'nrhs_ratio_max', 'nrhs_ratio_min', 'ota_nn_max', 'ota_nn_min', 'ota_np_max', 'ota_np_min', 'ota_pn_max', 'ota_pn_min', 'ota_pp_max', 'ota_pp_min', 'prhs_ratio_max', 'prhs_ratio_min', 'ps_down', 'ps_product', 'ps_ratio', 'ps_sum', 'ps_up', 'root_cdeg_max', 'root_cdeg_mean', 'root_cdeg_min', 'root_cdeg_var', 'root_ncoefs_count', 'root_ncoefs_max', 'root_ncoefs_mean', 'root_ncoefs_min', 'root_ncoefs_var', 'root_pcoefs_count', 'root_pcoefs_max', 'root_pcoefs_mean', 'root_pcoefs_min', 'root_pcoefs_var', 'slack', 'solfracs']
FEATURE_NAMES = NODE_FEATURES + KHALI_FEATURES

class Macros:
    USE_ROOT_CDEG_MEAN = set(['ROOT_CDEG_MEAN', 'ROOT_CDEG_VAR', 'CDEG_MEAN_RATIO'])
    USE_ROOT_CDEG_MIN = set(['ROOT_CDEG_MIN', 'CDEG_MIN_RATIO'])
    USE_ROOT_CDEG_MAX = set(['ROOT_CDEG_MAX', 'CDEG_MAX_RATIO'])
    USE_ROOT = set(['TYPE_0', 'TYPE_1', 'TYPE_2', 'TYPE_3', 'COEFS', 'COEFS_NEG', 'COEFS_POS', 'NNZRS', 'ROOT_CDEG_MAX', 'ROOT_CDEG_MEAN', 'ROOT_CDEG_MIN', 'ROOT_CDEG_VAR', 'ROOT_NCOEFS_COUNT', 'ROOT_NCOEFS_MAX', 'ROOT_NCOEFS_MEAN', 'ROOT_NCOEFS_MIN', 'ROOT_NCOEFS_VAR', 'ROOT_PCOEFS_COUNT', 'ROOT_PCOEFS_MAX', 'ROOT_PCOEFS_MEAN', 'ROOT_PCOEFS_MIN', 'ROOT_PCOEFS_VAR'])
    USE_TYPE = set(['TYPE_0', 'TYPE_1', 'TYPE_2', 'TYPE_3'])

    USE_ROOT_CDEG = set(['ROOT_CDEG_MEAN', 'ROOT_CDEG_VAR', 'ROOT_CDEG_MIN', 'ROOT_CDEG_MAX'])
    USE_ROOT_COLUMN_PCOEFS = set(['ROOT_PCOEFS_MEAN', 'ROOT_PCOEFS_VAR', 'ROOT_PCOEFS_COUNT', 'ROOT_PCOEFS_MAX', 'ROOT_PCOEFS_MIN'])
    USE_ROOT_COLUMN_NCOEFS = set(['ROOT_NCOEFS_MEAN', 'ROOT_NCOEFS_VAR', 'ROOT_NCOEFS_COUNT', 'ROOT_NCOEFS_MAX', 'ROOT_NCOEFS_MIN'])
    UNION_USE_ROOT_COLUMN = set(['USE_ROOT_CDEG', 'USE_ROOT_COLUMN_PCOEFS', 'USE_ROOT_COLUMN_NCOEFS'])

    USE_ACTIVE1 = set(['ACONS_NB1', 'ACONS_SUM1', 'ACONS_MEAN1', 'ACONS_VAR1', 'ACONS_MAX1', 'ACONS_MIN1'])
    USE_ACTIVE2 = set(['ACONS_NB2', 'ACONS_SUM2', 'ACONS_MEAN2', 'ACONS_VAR2', 'ACONS_MAX2', 'ACONS_MIN2'])
    USE_ACTIVE3 = set(['ACONS_NB3', 'ACONS_SUM3', 'ACONS_MEAN3', 'ACONS_VAR3', 'ACONS_MAX3', 'ACONS_MIN3'])
    USE_ACTIVE4 = set(['ACONS_NB4', 'ACONS_SUM4', 'ACONS_MEAN4', 'ACONS_VAR4', 'ACONS_MAX4', 'ACONS_MIN4'])
    USE_ACONS_SUM1 =set(['ACONS_SUM1', 'ACONS_VAR1'])
    USE_ACONS_SUM2 =set(['ACONS_SUM2', 'ACONS_VAR2'])
    USE_ACONS_SUM3 =set(['ACONS_SUM3', 'ACONS_VAR3'])
    USE_ACONS_SUM4 =set(['ACONS_SUM4', 'ACONS_VAR4'])
    USE_ACONS_MEAN1 = set(['ACONS_MEAN1', 'ACONS_VAR1'])
    USE_ACONS_MEAN2 = set(['ACONS_MEAN2', 'ACONS_VAR2'])
    USE_ACONS_MEAN3 = set(['ACONS_MEAN3', 'ACONS_VAR3'])
    USE_ACONS_MEAN4 = set(['ACONS_MEAN4', 'ACONS_VAR4'])
    UNION_USE_ACTIVE = set(['USE_ACTIVE1', 'USE_ACTIVE2', 'USE_ACTIVE3', 'USE_ACTIVE4'])

    USE_LB = set(["HAS_LB", "SOL_IS_AT_LB"])
    USE_UB = set(["HAS_UB", "SOL_IS_AT_UB"])
    USE_BASIS_STATUS = set(['BASIS_STATUS_0', 'BASIS_STATUS_1', 'BASIS_STATUS_2', 'BASIS_STATUS_3'])
    USE_PSEUDO = set(['PS_DOWN', 'PS_PRODUCT', 'PS_RATIO', 'PS_SUM', 'PS_UP'])

    UNION_USE_COLUMN = set(["USE_CDEG", "USE_RHS", "USE_OTA", "USE_ACTIVE"])

    USE_CDEG_MEAN = set(['CDEG_MEAN', 'CDEG_VAR', 'CDEG_MEAN_RATIO'])
    USE_CDEG_MAX = set(['CDEG_MAX', 'CDEG_MAX_RATIO'])
    USE_CDEG_MIN = set(['CDEG_MIN', 'CDEG_MIN_RATIO'])
    USE_CDEG = set(['CDEG_MIN', 'CDEG_MAX', 'CDEG_MEAN', 'CDEG_VAR', 'CDEG_MEAN_RATIO', 'CDEG_MAX_RATIO', 'CDEG_MIN_RATIO'])

    USE_RHS = set(['PRHS_RATIO_MAX', 'PRHS_RATIO_MIN', 'NRHS_RATIO_MAX', 'NRHS_RATIO_MIN'])

    USE_OTA_P = set(['OTA_PN_MAX', 'OTA_PN_MIN', 'OTA_PP_MAX', 'OTA_PP_MIN'])
    USE_OTA_N = set(['OTA_NN_MAX', 'OTA_NN_MIN', 'OTA_NP_MAX', 'OTA_NP_MIN'])
    UNION_USE_OTA = set(['USE_OTA_P', 'USE_OTA_N'])

    USE_COEFS = set(['COEF_NORMALIZED', 'COEFS'])

    UNION_SEQ = ['UNION_USE_ROOT_COLUMN', 'UNION_USE_ACTIVE', 'UNION_USE_OTA', 'UNION_USE_COLUMN']

COLUMN_FEATURES = Macros.USE_CDEG | Macros.USE_RHS | Macros.USE_OTA_P | Macros.USE_OTA_N | Macros.USE_ACTIVE1 | Macros.USE_ACTIVE2 | Macros.USE_ACTIVE3 | Macros.USE_ACTIVE4
COLUMN_FEATURES = set( x.lower() for x in COLUMN_FEATURES)
assert COLUMN_FEATURES.issubset(set(FEATURE_NAMES))

MATH_ARITY = OrderedDict([ # arity 2 -> arity 1
    ('+',2),
    ('-',2),
    ('*',2),
    ('/',2), 
    ('^', 2),
    ('exp',1),
    ('log',1),
])

CONSTANT_OPERATORS = ["2.0", "5.0", "10.0", "0.1", "0.2", "0.5"] # @ means a place holder which will be optimized in the inner loop
INVERSE_OPERATOR_DICT = {"exp": "log", "log": "exp", "sqrt": "square", "square": "sqrt"}

def binary_f(x,a,b):
    return f"({a} {x} {b})"
def unary_f(x,a):
    return f"torch.{x}({a})"
def unary_f_nlp(x,a):
    return f"{x}({a})"
def nlp_power(x,a,b):
    return f"{a}^{b}"
def nlp_square(x,a):
    return f"{a}^2"
def nlp_sqrt(x,a):
    return f"{a}^0.5"

def power(x,a,b):
    return f"torch.pow({a}, {b})"

TORCH_OPERATOR_DICT = {
    "^": power,
}

NLP_OPERATOR_DICT = {
    "^": nlp_power,
}


class Operators:

    def __init__(self, const_list=None, math_list="all", var_list="full"):
        """
        order: vars, consts, arity_two_operators, arity_one_operators
        """
        self.var_operators = FEATURE_NAMES
        if var_list=="simple":
            column_var_mask = [i for i, name in enumerate(FEATURE_NAMES) if name in COLUMN_FEATURES]
            self.column_var_mask = torch.tensor(column_var_mask, dtype=torch.long)
            assert len(self.column_var_mask) == 91 - 48
        else:
            assert var_list=="full"
            self.column_var_mask = None

        if const_list is None:
            self.constant_operators = CONSTANT_OPERATORS[:]
        else:
            self.constant_operators = const_list
        if math_list == "simple":
            self.math_operators = ['+', '-', '*']
        elif math_list == "all":
            self.math_operators = list(MATH_ARITY.keys())
        elif type(math_list) is list:
            math_set = set(math_list)
            self.math_operators = [x for x in MATH_ARITY if x in math_set]
        else:
            raise NotImplementedError(f"wrong math operators list {"!"*10}")

        self.operator_list = self.var_operators + self.constant_operators + self.math_operators
        self.operator_dict = {k:i for i,k in enumerate(self.operator_list)}
        self.operator_length = len(self.operator_list)

        arity_dict = defaultdict(int, MATH_ARITY)
        self.arity_list = [arity_dict[operator] for operator in self.operator_list]
        self.arity_tensor = torch.tensor(self.arity_list, dtype=torch.long)

        self.zero_arity_mask = torch.tensor([True if arity_dict[x]==0 else False for x in self.operator_list], dtype=torch.bool)[None, :]
        self.nonzero_arity_mask = torch.tensor([True if arity_dict[x]!=0 else False for x in self.operator_list], dtype=torch.bool)[None, :]

        self.have_inverse = torch.tensor([((operator in INVERSE_OPERATOR_DICT) and (INVERSE_OPERATOR_DICT[operator] in self.operator_dict)) for operator in self.operator_list], dtype=torch.bool)
        self.where_inverse = torch.full(size=(self.operator_length,), fill_value=int(1e5), dtype=torch.long)
        self.where_inverse[self.have_inverse] = torch.tensor([self.operator_dict[INVERSE_OPERATOR_DICT[operator]] for i, operator in enumerate(self.operator_list) if self.have_inverse[i]], dtype=torch.long)

        variable_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        variable_mask[:len(self.var_operators)] = True
        const_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        const_mask[len(self.var_operators):-len(self.math_operators)] = True
        self.variable_mask = variable_mask[None, :]
        self.non_variable_mask = torch.logical_not(self.variable_mask)
        self.const_mask = const_mask[None, :]
        self.non_const_mask = torch.logical_not(self.const_mask)

        num_math_arity_two = sum([1 for x in self.math_operators if MATH_ARITY[x]==2])
        num_math_arity_one = len(self.math_operators) - num_math_arity_two
        self.arity_zero_begin, self.arity_zero_end = 0, len(self.var_operators) + len(self.constant_operators)
        self.arity_two_begin, self.arity_two_end = len(self.var_operators) + len(self.constant_operators), len(self.var_operators) + len(self.constant_operators) + num_math_arity_two
        self.arity_one_begin, self.arity_one_end = len(self.operator_list) - num_math_arity_one, len(self.operator_list)

        self.variable_begin, self.variable_end = 0, len(self.var_operators)


    def is_var_i(self, i):
        return self.variable_begin <= i < self.variable_end
