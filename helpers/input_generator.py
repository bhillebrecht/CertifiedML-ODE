from pyDOE import lhs

def generate_collocation_points(N_phys, lb, ub):
    """
    Generates collocation points for PINN training.

    :param int N_phys: number of points generated
    :param np.array lb: 1D array of lower bounds, length must agree with ub
    :param np.array ub: 1D array of upper bounds, length determines dimensionality of returned array to (N_phys, ub.len)
    """
    X_phys = lb + (ub - lb) * lhs(len(ub), N_phys)
    return X_phys
