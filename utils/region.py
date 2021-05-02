import numpy as np

def normalize(vec):

    return vec / np.linalg.norm(vec)

def find_cube_from_vector(a, b, square_diag=10):

    a_to_b = b - a
    a_to_b_norm = np.linalg.norm(a_to_b)
    a_to_b_unit = a_to_b / a_to_b_norm

    start, end = a + a_to_b_unit * (a_to_b_norm * .25), a + a_to_b_unit * (a_to_b_norm * .75)

    vectors = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]

    chosen_v = None

    for vector in vectors:

        if np.linalg.norm(np.cross(vector, a_to_b_unit)) > 0.0:
            chosen_v = vector
            break

    u1 = np.cross(a_to_b_unit, chosen_v)

    assert np.dot(u1, a_to_b_unit) == 0.0

    u1 = normalize(u1)

    u2 = np.cross(a_to_b_unit, u1)
    u2 = normalize(u2)

    square_half_diag = square_diag / 2.

    p1 = start + u1 * square_half_diag
    p2 = start + u2 * square_half_diag
    p3 = start - u1 * square_half_diag
    p4 = start - u2 * square_half_diag

    p5 = end + u1 * square_half_diag
    p6 = end + u2 * square_half_diag
    p7 = end - u1 * square_half_diag
    p8 = end - u2 * square_half_diag

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    points = [p1, p2, p3, p4, p5, p6, p7, p8]

    return points, edges