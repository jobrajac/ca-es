from numba import njit

@njit
def hebbian_update(heb_coeffs, weights1_2, weights2_3, o0, o1, o2):
    """Update the weights of two network layers using Hebbian coefficients and pre- and postsynaptic values. ABCD-rule."""
    heb_offset = 0
    for k in range(o0.shape[1]):
        for l in range(o0.shape[2]):
            o0_temp = o0[0][k][l]  # (24)
            o1_temp = o1[0][k][l]  # (32)
            for i in range(weights1_2.shape[1]):
                for j in range(weights1_2.shape[0]):
                    idx = (weights1_2.shape[0] - 1) * i + i + j
                    updt = heb_coeffs[idx][0] * o0_temp[i] * o1_temp[j] + heb_coeffs[idx][1] * o0_temp[i] + \
                           heb_coeffs[idx][2] * o1_temp[j] + heb_coeffs[idx][4]
                    weights1_2[:, i][j] += heb_coeffs[idx][3] * updt

    heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
    # Layer 2
    for k in range(o1.shape[1]):
        for l in range(o1.shape[2]):
            o1_temp = o1[0][k][l]
            o2_temp = o2[0][k][l]
            for i in range(weights2_3.shape[1]):
                for j in range(weights2_3.shape[0]):
                    idx = heb_offset + (weights2_3.shape[0] - 1) * i + i + j
                    weights2_3[:, i][j] += heb_coeffs[idx][3] * (heb_coeffs[idx][0] * o1_temp[i] * o2_temp[j]
                                                                 + heb_coeffs[idx][1] * o1_temp[i]
                                                                 + heb_coeffs[idx][2] * o2_temp[j] + heb_coeffs[idx][4])

    return weights1_2, weights2_3