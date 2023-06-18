"""Test partial_trace."""
import cvxpy
import numpy as np

from cvxpy.atoms.affine.vstack import Vstack
from toqito.channels import partial_trace


def test_partial_trace():
    """
    Standard call to partial_trace.

    By default, the partial_trace function takes the trace over the second
    subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[7, 11], [23, 27]])

    res = partial_trace(test_input_mat)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys():
    """
    Specify the `sys` argument.

    By specifying the `sys` argument, you can perform the partial trace
    the first subsystem instead:
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[12, 14], [20, 22]])

    res = partial_trace(test_input_mat, [0])

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys_int():
    """Supply `sys` argument as int."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[12, 14], [20, 22]])

    res = partial_trace(test_input_mat, 0)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys_int_dim_int():
    """
    Default second subsystem.

    By default, the partial_transpose function takes the trace over
    the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[7, 11], [23, 27]])

    res = partial_trace(test_input_mat, [1])

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys_int_dim_int_2():
    """
    Default second subsystem.

    By default, the partial_transpose function takes the trace over
    the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = 34

    res = partial_trace(test_input_mat, [1], [1, 4])

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_4_by_4():
    """Test for 4-by-4 matrix."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    pt_1 = partial_trace(test_input_mat, [0], [2, 2])
    pt_2 = partial_trace(test_input_mat, [1], [2, 2])

    expected_pt_1 = np.array([[12, 14], [20, 22]])
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    expected_pt_2 = np.array([[7, 11], [23, 27]])
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_8_by_8():
    """Test for 8-by-8 matrix."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [2, 2, 2])
    expected_pt_1 = np.array(
        [[38, 40, 42, 44], [54, 56, 58, 60], [70, 72, 74, 76], [86, 88, 90, 92]]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [2, 2, 2])
    expected_pt_2 = np.array(
        [[20, 22, 28, 30], [36, 38, 44, 46], [84, 86, 92, 94], [100, 102, 108, 110]]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out third subsystem:
    pt_3 = partial_trace(test_input_mat, [2], [2, 2, 2])
    expected_pt_3 = np.array(
        [[11, 15, 19, 23], [43, 47, 51, 55], [75, 79, 83, 87], [107, 111, 115, 119]]
    )
    bool_mat = np.isclose(expected_pt_3, pt_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and second subsystem:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [2, 2, 2])
    expected_pt_1_2 = np.array([[112, 116], [144, 148]])
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and third subsystem:
    pt_1_3 = partial_trace(test_input_mat, [0, 2], [2, 2, 2])
    expected_pt_1_3 = np.array([[94, 102], [158, 166]])
    bool_mat = np.isclose(expected_pt_1_3, pt_1_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second and third subsystem:
    pt_2_3 = partial_trace(test_input_mat, [1, 2], [2, 2, 2])
    expected_pt_2_3 = np.array([[58, 74], [186, 202]])
    bool_mat = np.isclose(expected_pt_2_3, pt_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_6_by_6_subsystems_2_3():
    """Test for 6-by-6 matrix for subsystems 2 x 3"""
    test_input_mat = np.arange(1, 37).reshape(6, 6)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [2, 3])
    expected_pt_1 = np.array([[23, 25, 27], [35, 37, 39], [47, 49, 51]])
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [2, 3])
    expected_pt_2 = np.array([[24, 33], [78, 87]])
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and second subsystem:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [2, 3])
    expected_pt_1_2 = np.array([[111]])
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_6_by_6_subsystems_3_2():
    """Test for 6-by-6 matrix for subsystems 3 x 2"""
    test_input_mat = np.arange(1, 37).reshape(6, 6)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [3, 2])
    expected_pt_1 = np.array([[45, 48], [63, 66]])
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [3, 2])
    expected_pt_2 = np.array([[9, 13, 17], [33, 37, 41], [57, 61, 65]])
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and second subsystem:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [3, 2])
    expected_pt_1_2 = np.array([[111]])
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_9_by_9_subsystems_3_3():
    """Test for 9-by-9 matrix for subsystems 3 x 3"""
    test_input_mat = np.arange(1, 82).reshape(9, 9)

    # Partial trace on first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [3, 3])
    expected_pt_1 = np.array([[93, 96, 99], [120, 123, 126], [147, 150, 153]])
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial trace on second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [3, 3])
    expected_pt_2 = np.array([[33, 42, 51], [114, 123, 132], [195, 204, 213]])
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial trace on first and second subsystems:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [3, 3])
    expected_pt_1_2 = np.array([[369]])
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_16_by_16_subsystems_2_2_2_2():
    """Test for 16-by-16 matrix for subsystems (2 x 2) x (2 x 2)."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [2, 2, 2, 2])
    expected_pt_1 = np.array(
        [
            [138, 140, 142, 144, 146, 148, 150, 152],
            [170, 172, 174, 176, 178, 180, 182, 184],
            [202, 204, 206, 208, 210, 212, 214, 216],
            [234, 236, 238, 240, 242, 244, 246, 248],
            [266, 268, 270, 272, 274, 276, 278, 280],
            [298, 300, 302, 304, 306, 308, 310, 312],
            [330, 332, 334, 336, 338, 340, 342, 344],
            [362, 364, 366, 368, 370, 372, 374, 376],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [2, 2, 2, 2])
    expected_pt_2 = np.array(
        [
            [70, 72, 74, 76, 86, 88, 90, 92],
            [102, 104, 106, 108, 118, 120, 122, 124],
            [134, 136, 138, 140, 150, 152, 154, 156],
            [166, 168, 170, 172, 182, 184, 186, 188],
            [326, 328, 330, 332, 342, 344, 346, 348],
            [358, 360, 362, 364, 374, 376, 378, 380],
            [390, 392, 394, 396, 406, 408, 410, 412],
            [422, 424, 426, 428, 438, 440, 442, 444],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out third subsystem:
    pt_3 = partial_trace(test_input_mat, [2], [2, 2, 2, 2])
    expected_pt_3 = np.array(
        [
            [36, 38, 44, 46, 52, 54, 60, 62],
            [68, 70, 76, 78, 84, 86, 92, 94],
            [164, 166, 172, 174, 180, 182, 188, 190],
            [196, 198, 204, 206, 212, 214, 220, 222],
            [292, 294, 300, 302, 308, 310, 316, 318],
            [324, 326, 332, 334, 340, 342, 348, 350],
            [420, 422, 428, 430, 436, 438, 444, 446],
            [452, 454, 460, 462, 468, 470, 476, 478],
        ]
    )
    bool_mat = np.isclose(expected_pt_3, pt_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out fourth subsystem:
    pt_4 = partial_trace(test_input_mat, [3], [2, 2, 2, 2])
    expected_pt_4 = np.array(
        [
            [19, 23, 27, 31, 35, 39, 43, 47],
            [83, 87, 91, 95, 99, 103, 107, 111],
            [147, 151, 155, 159, 163, 167, 171, 175],
            [211, 215, 219, 223, 227, 231, 235, 239],
            [275, 279, 283, 287, 291, 295, 299, 303],
            [339, 343, 347, 351, 355, 359, 363, 367],
            [403, 407, 411, 415, 419, 423, 427, 431],
            [467, 471, 475, 479, 483, 487, 491, 495],
        ]
    )
    bool_mat = np.isclose(expected_pt_4, pt_4)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and second subsystem:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [2, 2, 2, 2])
    expected_pt_1_2 = np.array(
        [
            [412, 416, 420, 424],
            [476, 480, 484, 488],
            [540, 544, 548, 552],
            [604, 608, 612, 616],
        ]
    )

    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and third subsystem:
    pt_1_3 = partial_trace(test_input_mat, [0, 2], [2, 2, 2, 2])
    expected_pt_1_3 = np.array(
        [
            [344, 348, 360, 364],
            [408, 412, 424, 428],
            [600, 604, 616, 620],
            [664, 668, 680, 684],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_3, pt_1_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and fourth subsystem:
    pt_1_4 = partial_trace(test_input_mat, [0, 3], [2, 2, 2, 2])

    expected_pt_1_4 = np.array(
        [[310, 318, 326, 334], [438, 446, 454, 462], [566, 574, 582, 590], [694, 702, 710, 718]]
    )
    bool_mat = np.isclose(expected_pt_1_4, pt_1_4)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second and third subsystem:
    pt_2_3 = partial_trace(test_input_mat, [1, 2], [2, 2, 2, 2])
    expected_pt_2_3 = np.array(
        [[208, 212, 240, 244], [272, 276, 304, 308], [720, 724, 752, 756], [784, 788, 816, 820]]
    )
    bool_mat = np.isclose(expected_pt_2_3, pt_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second and fourth subsystem:
    pt_2_4 = partial_trace(test_input_mat, [1, 3], [2, 2, 2, 2])
    expected_pt_2_4 = np.array(
        [[174, 182, 206, 214], [302, 310, 334, 342], [686, 694, 718, 726], [814, 822, 846, 854]]
    )
    bool_mat = np.isclose(expected_pt_2_4, pt_2_4)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out third and fourth subsystem:
    pt_3_4 = partial_trace(test_input_mat, [2, 3], [2, 2, 2, 2])
    expected_pt_3_4 = np.array(
        [[106, 122, 138, 154], [362, 378, 394, 410], [618, 634, 650, 666], [874, 890, 906, 922]]
    )
    bool_mat = np.isclose(expected_pt_3_4, pt_3_4)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_16_by_16_subsystems_2_2_4():
    """Test for 16-by-16 matrix for subsystems 2 x 4."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [2, 2, 4])
    expected_pt_1 = np.array(
        [
            [138, 140, 142, 144, 146, 148, 150, 152],
            [170, 172, 174, 176, 178, 180, 182, 184],
            [202, 204, 206, 208, 210, 212, 214, 216],
            [234, 236, 238, 240, 242, 244, 246, 248],
            [266, 268, 270, 272, 274, 276, 278, 280],
            [298, 300, 302, 304, 306, 308, 310, 312],
            [330, 332, 334, 336, 338, 340, 342, 344],
            [362, 364, 366, 368, 370, 372, 374, 376],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [2, 2, 4])
    expected_pt_2 = np.array(
        [
            [70, 72, 74, 76, 86, 88, 90, 92],
            [102, 104, 106, 108, 118, 120, 122, 124],
            [134, 136, 138, 140, 150, 152, 154, 156],
            [166, 168, 170, 172, 182, 184, 186, 188],
            [326, 328, 330, 332, 342, 344, 346, 348],
            [358, 360, 362, 364, 374, 376, 378, 380],
            [390, 392, 394, 396, 406, 408, 410, 412],
            [422, 424, 426, 428, 438, 440, 442, 444],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out third subsystem:
    pt_3 = partial_trace(test_input_mat, [2], [2, 2, 4])
    expected_pt_3 = np.array(
        [[106, 122, 138, 154], [362, 378, 394, 410], [618, 634, 650, 666], [874, 890, 906, 922]]
    )
    bool_mat = np.isclose(expected_pt_3, pt_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and second subsystem:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [2, 2, 4])
    expected_pt_1_2 = np.array(
        [
            [412, 416, 420, 424],
            [476, 480, 484, 488],
            [540, 544, 548, 552],
            [604, 608, 612, 616],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and third subsystem:
    pt_1_3 = partial_trace(test_input_mat, [0, 2], [2, 2, 4])
    expected_pt_1_3 = np.array([[756, 788], [1268, 1300]])
    bool_mat = np.isclose(expected_pt_1_3, pt_1_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second and third subsystem:
    pt_2_3 = partial_trace(test_input_mat, [1, 2], [2, 2, 4])
    expected_pt_2_3 = np.array([[484, 548], [1508, 1572]])
    bool_mat = np.isclose(expected_pt_2_3, pt_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_16_by_16_subsystems_4_2_2():
    """Test for 16-by-16 matrix for subsystems 4 x 2."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)

    # Trace out the first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [4, 2, 2])
    expected_pt_1 = np.array(
        [
            [412, 416, 420, 424],
            [476, 480, 484, 488],
            [540, 544, 548, 552],
            [604, 608, 612, 616],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out the second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [4, 2, 2])
    expected_pt_2 = np.array(
        [
            [36, 38, 44, 46, 52, 54, 60, 62],
            [68, 70, 76, 78, 84, 86, 92, 94],
            [164, 166, 172, 174, 180, 182, 188, 190],
            [196, 198, 204, 206, 212, 214, 220, 222],
            [292, 294, 300, 302, 308, 310, 316, 318],
            [324, 326, 332, 334, 340, 342, 348, 350],
            [420, 422, 428, 430, 436, 438, 444, 446],
            [452, 454, 460, 462, 468, 470, 476, 478],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out the third subsystem:
    pt_3 = partial_trace(test_input_mat, [2], [4, 2, 2])
    expected_pt_3 = np.array(
        [
            [19, 23, 27, 31, 35, 39, 43, 47],
            [83, 87, 91, 95, 99, 103, 107, 111],
            [147, 151, 155, 159, 163, 167, 171, 175],
            [211, 215, 219, 223, 227, 231, 235, 239],
            [275, 279, 283, 287, 291, 295, 299, 303],
            [339, 343, 347, 351, 355, 359, 363, 367],
            [403, 407, 411, 415, 419, 423, 427, 431],
            [467, 471, 475, 479, 483, 487, 491, 495],
        ]
    )
    bool_mat = np.isclose(expected_pt_3, pt_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out the first and second subsystems:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [4, 2, 2])
    expected_pt_1_2 = np.array([[960, 968], [1088, 1096]])
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out the first and third subsystems:
    pt_1_3 = partial_trace(test_input_mat, [0, 2], [4, 2, 2])
    expected_pt_1_3 = np.array([[892, 908], [1148, 1164]])
    bool_mat = np.isclose(expected_pt_1_3, pt_1_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out the second and third subsystems:
    pt_2_3 = partial_trace(test_input_mat, [1, 2], [4, 2, 2])
    expected_pt_2_3 = np.array(
        [[106, 122, 138, 154], [362, 378, 394, 410], [618, 634, 650, 666], [874, 890, 906, 922]]
    )
    bool_mat = np.isclose(expected_pt_2_3, pt_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_16_by_16_subsystems_4_4():
    """Test for 16-by-16 matrix for subsystems 4 x 4."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [4, 4])
    expected_pt_1 = np.array(
        [[412, 416, 420, 424], [476, 480, 484, 488], [540, 544, 548, 552], [604, 608, 612, 616]]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:
    pt_2 = partial_trace(test_input_mat, [1], [4, 4])
    expected_pt_2 = np.array(
        [[106, 122, 138, 154], [362, 378, 394, 410], [618, 634, 650, 666], [874, 890, 906, 922]]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_64_by_64_subsystems_4_4_2_2():
    """Test for 64-by-64 matrix for subsystems 4 x 4 x 2 x 2."""
    test_input_mat = np.arange(1, 4097).reshape(64, 64)

    # Trace out first subsystem:
    # Trace out second subsystem:
    # Trace out third subsystem:
    # Trace out fourth subsystem:

    # Trace out first and second subsystem:
    # Trace out first and third subsystem:
    # Trace out first and fourth subsystem:

    # Trace out second and third subsystem:
    # Trace out second and fourth subsystem:

    # Trace out third and fourth subsystem:
    pt_3_4 = partial_trace(test_input_mat, [2, 3], [4, 4, 2, 2])
    expected_pt_3_4 = np.array(
        [
            [394, 410, 426, 442, 458, 474, 490, 506, 522, 538, 554, 570, 586, 602, 618, 634],
            [
                1418,
                1434,
                1450,
                1466,
                1482,
                1498,
                1514,
                1530,
                1546,
                1562,
                1578,
                1594,
                1610,
                1626,
                1642,
                1658,
            ],
            [
                2442,
                2458,
                2474,
                2490,
                2506,
                2522,
                2538,
                2554,
                2570,
                2586,
                2602,
                2618,
                2634,
                2650,
                2666,
                2682,
            ],
            [
                3466,
                3482,
                3498,
                3514,
                3530,
                3546,
                3562,
                3578,
                3594,
                3610,
                3626,
                3642,
                3658,
                3674,
                3690,
                3706,
            ],
            [
                4490,
                4506,
                4522,
                4538,
                4554,
                4570,
                4586,
                4602,
                4618,
                4634,
                4650,
                4666,
                4682,
                4698,
                4714,
                4730,
            ],
            [
                5514,
                5530,
                5546,
                5562,
                5578,
                5594,
                5610,
                5626,
                5642,
                5658,
                5674,
                5690,
                5706,
                5722,
                5738,
                5754,
            ],
            [
                6538,
                6554,
                6570,
                6586,
                6602,
                6618,
                6634,
                6650,
                6666,
                6682,
                6698,
                6714,
                6730,
                6746,
                6762,
                6778,
            ],
            [
                7562,
                7578,
                7594,
                7610,
                7626,
                7642,
                7658,
                7674,
                7690,
                7706,
                7722,
                7738,
                7754,
                7770,
                7786,
                7802,
            ],
            [
                8586,
                8602,
                8618,
                8634,
                8650,
                8666,
                8682,
                8698,
                8714,
                8730,
                8746,
                8762,
                8778,
                8794,
                8810,
                8826,
            ],
            [
                9610,
                9626,
                9642,
                9658,
                9674,
                9690,
                9706,
                9722,
                9738,
                9754,
                9770,
                9786,
                9802,
                9818,
                9834,
                9850,
            ],
            [
                10634,
                10650,
                10666,
                10682,
                10698,
                10714,
                10730,
                10746,
                10762,
                10778,
                10794,
                10810,
                10826,
                10842,
                10858,
                10874,
            ],
            [
                11658,
                11674,
                11690,
                11706,
                11722,
                11738,
                11754,
                11770,
                11786,
                11802,
                11818,
                11834,
                11850,
                11866,
                11882,
                11898,
            ],
            [
                12682,
                12698,
                12714,
                12730,
                12746,
                12762,
                12778,
                12794,
                12810,
                12826,
                12842,
                12858,
                12874,
                12890,
                12906,
                12922,
            ],
            [
                13706,
                13722,
                13738,
                13754,
                13770,
                13786,
                13802,
                13818,
                13834,
                13850,
                13866,
                13882,
                13898,
                13914,
                13930,
                13946,
            ],
            [
                14730,
                14746,
                14762,
                14778,
                14794,
                14810,
                14826,
                14842,
                14858,
                14874,
                14890,
                14906,
                14922,
                14938,
                14954,
                14970,
            ],
            [
                15754,
                15770,
                15786,
                15802,
                15818,
                15834,
                15850,
                15866,
                15882,
                15898,
                15914,
                15930,
                15946,
                15962,
                15978,
                15994,
            ],
        ]
    )
    bool_mat = np.isclose(expected_pt_3_4, pt_3_4)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first, second, and third subsystem:
    pt_1_2_3 = partial_trace(test_input_mat, [0, 1, 2], [4, 4, 2, 2])
    expected_pt_1_2_3 = np.array([[64512, 64544], [66560, 66592]])
    bool_mat = np.isclose(expected_pt_1_2_3, pt_1_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first, third, and fourth subsystem:
    pt_1_3_4 = partial_trace(test_input_mat, [0, 2, 3], [4, 4, 2, 2])
    expected_pt_1_3_4 = np.array(
        [
            [26536, 26600, 26664, 26728],
            [30632, 30696, 30760, 30824],
            [34728, 34792, 34856, 34920],
            [38824, 38888, 38952, 39016],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_3_4, pt_1_3_4)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first, second, third, and fourth subsystem:
    pt_1_2_3_4 = partial_trace(test_input_mat, [0, 1, 2, 3], [4, 4, 2, 2])
    expected_pt_1_2_3_4 = np.array([[131104]])
    bool_mat = np.isclose(expected_pt_1_2_3_4, pt_1_2_3_4)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_64_by_64_subsystems_4_4_4():
    """Test for 64-by-64 matrix for subsystems 4 x 4 x 4."""
    test_input_mat = np.arange(1, 4097).reshape(64, 64)

    # Trace out first subsystem:
    pt_1 = partial_trace(test_input_mat, [0], [4, 4, 4])
    expected_pt_1 = np.array(
        [
            [
                6244,
                6248,
                6252,
                6256,
                6260,
                6264,
                6268,
                6272,
                6276,
                6280,
                6284,
                6288,
                6292,
                6296,
                6300,
                6304,
            ],
            [
                6500,
                6504,
                6508,
                6512,
                6516,
                6520,
                6524,
                6528,
                6532,
                6536,
                6540,
                6544,
                6548,
                6552,
                6556,
                6560,
            ],
            [
                6756,
                6760,
                6764,
                6768,
                6772,
                6776,
                6780,
                6784,
                6788,
                6792,
                6796,
                6800,
                6804,
                6808,
                6812,
                6816,
            ],
            [
                7012,
                7016,
                7020,
                7024,
                7028,
                7032,
                7036,
                7040,
                7044,
                7048,
                7052,
                7056,
                7060,
                7064,
                7068,
                7072,
            ],
            [
                7268,
                7272,
                7276,
                7280,
                7284,
                7288,
                7292,
                7296,
                7300,
                7304,
                7308,
                7312,
                7316,
                7320,
                7324,
                7328,
            ],
            [
                7524,
                7528,
                7532,
                7536,
                7540,
                7544,
                7548,
                7552,
                7556,
                7560,
                7564,
                7568,
                7572,
                7576,
                7580,
                7584,
            ],
            [
                7780,
                7784,
                7788,
                7792,
                7796,
                7800,
                7804,
                7808,
                7812,
                7816,
                7820,
                7824,
                7828,
                7832,
                7836,
                7840,
            ],
            [
                8036,
                8040,
                8044,
                8048,
                8052,
                8056,
                8060,
                8064,
                8068,
                8072,
                8076,
                8080,
                8084,
                8088,
                8092,
                8096,
            ],
            [
                8292,
                8296,
                8300,
                8304,
                8308,
                8312,
                8316,
                8320,
                8324,
                8328,
                8332,
                8336,
                8340,
                8344,
                8348,
                8352,
            ],
            [
                8548,
                8552,
                8556,
                8560,
                8564,
                8568,
                8572,
                8576,
                8580,
                8584,
                8588,
                8592,
                8596,
                8600,
                8604,
                8608,
            ],
            [
                8804,
                8808,
                8812,
                8816,
                8820,
                8824,
                8828,
                8832,
                8836,
                8840,
                8844,
                8848,
                8852,
                8856,
                8860,
                8864,
            ],
            [
                9060,
                9064,
                9068,
                9072,
                9076,
                9080,
                9084,
                9088,
                9092,
                9096,
                9100,
                9104,
                9108,
                9112,
                9116,
                9120,
            ],
            [
                9316,
                9320,
                9324,
                9328,
                9332,
                9336,
                9340,
                9344,
                9348,
                9352,
                9356,
                9360,
                9364,
                9368,
                9372,
                9376,
            ],
            [
                9572,
                9576,
                9580,
                9584,
                9588,
                9592,
                9596,
                9600,
                9604,
                9608,
                9612,
                9616,
                9620,
                9624,
                9628,
                9632,
            ],
            [
                9828,
                9832,
                9836,
                9840,
                9844,
                9848,
                9852,
                9856,
                9860,
                9864,
                9868,
                9872,
                9876,
                9880,
                9884,
                9888,
            ],
            [
                10084,
                10088,
                10092,
                10096,
                10100,
                10104,
                10108,
                10112,
                10116,
                10120,
                10124,
                10128,
                10132,
                10136,
                10140,
                10144,
            ],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second subsystem:

    # Trace out third subsystem:

    # Trace out first and second subsystem:
    pt_1_2 = partial_trace(test_input_mat, [0, 1], [4, 4, 4])
    expected_pt_1_2 = np.array(
        [
            [31216, 31232, 31248, 31264],
            [32240, 32256, 32272, 32288],
            [33264, 33280, 33296, 33312],
            [34288, 34304, 34320, 34336],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first and third subsystem:
    pt_1_3 = partial_trace(test_input_mat, [0, 2], [4, 4, 4])
    expected_pt_1_3 = np.array(
        [
            [26536, 26600, 26664, 26728],
            [30632, 30696, 30760, 30824],
            [34728, 34792, 34856, 34920],
            [38824, 38888, 38952, 39016],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_3, pt_1_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out second and third subsystem:
    pt_2_3 = partial_trace(test_input_mat, [1, 2], [4, 4, 4])
    expected_pt_2_3 = np.array(
        [
            [7816, 8072, 8328, 8584],
            [24200, 24456, 24712, 24968],
            [40584, 40840, 41096, 41352],
            [56968, 57224, 57480, 57736],
        ]
    )
    bool_mat = np.isclose(expected_pt_2_3, pt_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Trace out first, second, and third subsystems:
    pt_1_2_3 = partial_trace(test_input_mat, [0, 1, 2], [4, 4, 4])
    expected_pt_1_2_3 = np.array([[131104]])

    bool_mat = np.isclose(expected_pt_1_2_3, pt_1_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_invalid_sys_arg():
    """The `sys` argument must be either a list or int."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
        partial_trace(rho, "invalid_input")


def test_partial_trace_non_square_matrix_dim_2():
    """Matrix must be square for partial trace."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        partial_trace(rho, [1], [2])


def test_partial_trace_cvxpy():
    """Test partial trace on cvxpy objects."""
    x_var = cvxpy.Variable((4, 4), hermitian=True)
    x_pt = partial_trace(x_var)
    np.testing.assert_equal(isinstance(x_pt, Vstack), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
