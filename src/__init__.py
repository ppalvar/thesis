import numpy as np
from typing import Tuple, List, Union, Optional
import warnings


def fdct_wrapping_window(x: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the two halves of a C^inf compactly supported window function.

    This function generates left and right halves of a window function that is infinitely
    differentiable (C^inf) and compactly supported on the interval [0, 1]. The window is
    normalized so that the sum of squares of the left and right halves equals 1.

    Parameters
    ----------
    x : np.ndarray or float
        Vector or matrix of abscissae (x-coordinates), with relevant values in [0, 1].
        Values outside [0, 1] will result in 0 or 1 outputs as appropriate.

    Returns
    -------
    wl : np.ndarray
        Vector or matrix containing samples of the left half of the window (x ∈ [0, 1])
    wr : np.ndarray
        Vector or matrix containing samples of the right half of the window (x ∈ [0, 1])

    Notes
    -----
    The window function is constructed such that:
    - For x ≤ 0: wr = 1, wl = 0
    - For x ≥ 1: wl = 1, wr = 0
    - For 0 < x < 1: smooth transition following exp(1-1/(1-exp(1-1/x))) formulas
    - The window is normalized so that wl² + wr² = 1 everywhere

    Originally implemented in MATLAB by Laurent Demanet, 2004.
    Python version by Pedro P. Álvarez, 2025.
    """
    wr = np.zeros_like(x)
    wl = np.zeros_like(x)

    # Handle very small numbers (equivalent to MATLAB's 2^-52)
    x[np.abs(x) < 2**-52] = 0

    # Right window computation
    mask = (x > 0) & (x < 1)
    wr[mask] = np.exp(1 - 1.0 / (1 - np.exp(1 - 1.0 / x[mask])))
    wr[x <= 0] = 1

    # Left window computation
    wl[mask] = np.exp(1 - 1.0 / (1 - np.exp(1 - 1.0 / (1 - x[mask]))))
    wl[x >= 1] = 1

    # Normalization
    normalization = np.sqrt(wl**2 + wr**2)
    wr = wr / normalization
    wl = wl / normalization

    return wl, wr


def gdct_cos(x: Union[np.ndarray, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the two halves of a compactly supported cosine-based window function.

    This function generates left and right halves of a window function that uses cosine
    transitions and is compactly supported on the interval [0, 1]. The window is
    normalized so that the sum of squares of the left and right halves equals 1.

    Parameters
    ----------
    x : np.ndarray or float
        Vector or matrix of abscissae (x-coordinates), with relevant values in [0, 1].
        Values outside [0, 1] will result in 0 or 1 outputs as appropriate.

    Returns
    -------
    wl : np.ndarray
        Vector or matrix containing samples of the left half of the window (x ∈ [0, 1])
    wr : np.ndarray
        Vector or matrix containing samples of the right half of the window (x ∈ [0, 1])

    Notes
    -----
    The window function is constructed such that:
    - For x ≤ 0: wr = 1, wl = 0
    - For x ≥ 1: wl = 1, wr = 0
    - For 0 < x < 1: smooth cosine transition following (1 + cos(pi*x))/2 formulas
    - The window is normalized so that wl² + wr² = 1 everywhere

    Originally implemented in MATLAB by Haizhao Yang.
    Python version by Pedro P. Álvarez, 2025.
    """
    wr = np.zeros_like(x)
    wl = np.zeros_like(x)

    # Handle very small numbers (equivalent to MATLAB's 2^-52)
    x[np.abs(x) < 2**-52] = 0

    # Right window computation
    mask = (x > 0) & (x < 1)
    wr[mask] = (1 + np.cos(np.pi * x[mask])) / 2
    wr[x <= 0] = 1

    # Left window computation
    wl[mask] = (1 + np.cos(np.pi * (1 - x[mask]))) / 2
    wl[x >= 1] = 1

    # Normalization
    normalization = np.sqrt(wl**2 + wr**2)
    wr = wr / normalization
    wl = wl / normalization

    return wl, wr


def gdct_fwd_red(
    x: np.ndarray,
    is_real: bool = False,
    sz: Optional[Tuple[int, int]] = None,
    R_high: Optional[float] = None,
    R_low: Optional[float] = None,
    rad: float = 2.0,
    is_cos: bool = True,
    t_sc: float = 1.0 - 1.0 / 8.0,
    s_sc: float = 1.0 / 2.0 + 1.0 / 8.0,
    red: int = 1,
    wedge_length_coarse: int = 8,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    Generalized Discrete Curvelet Transform - Forward Transform with Redundancy

    Generates multi-frames by rotation in the frequency domain to create a
    redundant curvelet transform.

    Parameters
    ----------
    x : np.ndarray
        2D input array (N1 x N2 matrix)
    is_real : bool, optional
        Type of transform (False for complex, True for real-valued curvelets)
    sz : tuple of int, optional
        Sampling density in the image [sz1, sz2]
    R_high : float, optional
        Upper bound of interested spectrum range
    R_low : float, optional
        Lower bound of interested spectrum range
    rad : float, optional
        Parameter to adjust size of curvelet supports in frequency domain (0,2]
    is_cos : bool, optional
        Type of window function (False for C^inf, True for cosine window)
    t_sc : float, optional
        Scaling parameter for radius
    s_sc : float, optional
        Scaling parameter for angle
    red : int, optional
        Redundancy parameter
    wedge_length_coarse : int, optional
        Length of coarsest wedge (minimum 4)

    Returns
    -------
    C : list of lists of np.ndarray
        Curvelet coefficients (main coefficients)
    C1 : list of lists of np.ndarray
        First component coefficients (derivative in first dimension)
    C2 : list of lists of np.ndarray
        Second component coefficients (derivative in second dimension)

    Notes
    -----
    The output structure is organized as:
    C{j}{l,i}(k1,k2) where:
    - j: scale index (from coarsest to finest)
    - l: angle index (starts on right, increases counter-clockwise)
    - i: frame index (from 1 to red)
    - k1,k2: spatial positions

    Original MATLAB implementation by Haizhao Yang
    """

    # Compute FFT of input
    X = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x))) / np.sqrt(x.size)
    N1, N2 = X.shape

    # Create position vectors
    if N1 % 2 == 0:
        pN1 = N1 // 2
        pos_vec_1 = np.arange(-pN1, pN1)
    else:
        pN1 = (N1 - 1) // 2
        pos_vec_1 = np.arange(-pN1, pN1 + 1)

    if N2 % 2 == 0:
        pN2 = N2 // 2
        pos_vec_2 = np.arange(-pN2, pN2)
    else:
        pN2 = (N2 - 1) // 2
        pos_vec_2 = np.arange(-pN2, pN2 + 1)

    Pos_index_1, Pos_index_2 = np.meshgrid(pos_vec_1, pos_vec_2, indexing="ij")
    X1 = (2 * np.pi * 1j * Pos_index_1) * X
    X2 = (2 * np.pi * 1j * Pos_index_2) * X

    # Handle real-valued case
    if is_real:
        start1 = (N1 + 1) % 2
        end1 = (N1 - (N1 % 2)) // 2
        start2 = (N2 + 1) % 2
        end2 = (N2 - (N2 % 2)) // 2

        X[start1:, start2:end2] = 0
        X[start1:end1, (N2 + 1 + (N2 + 1) % 2) // 2] = 0

        X1[start1:, start2:end2] = 0
        X1[start1:end1, (N2 + 1 + (N2 + 1) % 2) // 2] = 0

        X2[start1:, start2:end2] = 0
        X2[start1:end1, (N2 + 1 + (N2 + 1) % 2) // 2] = 0

    # Parameter validation
    if wedge_length_coarse < 4:
        wedge_length_coarse = 4
        warnings.warn(
            "wedge_length_coarse is too small. Using wedge_length_coarse = 4."
        )

    # Initialize wedge parameters
    R = np.sqrt(N1**2 + N2**2) / 2
    if R_high is not None:
        R = min(R_high, R)

    if R_low is None:
        R_low = 0

    if R_low <= wedge_length_coarse:
        R_low = 0
        warnings.warn("R_low <= wedge_length_coarse, setting R_low = 0.")

    if rad > 2:
        rad = 2
        warnings.warn("rad too large! Setting rad = 2.")

    if R <= R_low:
        raise ValueError(f"R_low is too large! R_low must be at most {R}")

    # Initialize wedge structure
    wedge_length_coarse = wedge_length_coarse * rad
    wedge_length = [2]
    wedge_end = [R, R + 2]

    while wedge_end[0] > wedge_length_coarse or len(wedge_end) < 4:
        Length = rad * wedge_end[0] ** t_sc / 2
        wedge_length = [Length] + wedge_length
        wedge_end = [wedge_end[0] - Length] + wedge_end

    wedge_length = [wedge_end[0]] + wedge_length

    if R_low == 0:
        # Main processing when R_low == 0
        nbscales = len(wedge_end)

        temp = [a + b for a, b in zip(wedge_length[2:], wedge_length[1:-1])]
        arc_length = [t ** (s_sc / t_sc) for t in temp]
        nbangles = [
            round(np.pi * e / a / 2) * 4 for e, a in zip(wedge_end[2:], arc_length)
        ]
        nbangles = [1, 1] + nbangles
        angles = [2 * np.pi / n for n in nbangles]

        C = [[] for _ in range(nbscales - 1)]
        C1 = [[] for _ in range(nbscales - 1)]
        C2 = [[] for _ in range(nbscales - 1)]

        # Determine sampling grid size
        NN = max(
            [
                wedge_length[-2] + wedge_length[-3],
                np.sqrt(
                    wedge_end[-2] ** 2
                    + wedge_end[-4] ** 2
                    - 2 * wedge_end[-4] * wedge_end[-2] * np.cos(angles[-2])
                ),
                2 * wedge_end[-2] * np.sin(angles[-2]),
            ]
        )
        NN = int(np.ceil(NN / 4) * 4)

        if sz is None:
            sz = (NN, NN)
            warnings.warn("Sampling grid not assigned. Using smallest grid.")
        else:
            if sz[0] < NN or sz[1] < NN:
                warnings.warn(
                    "Sampling grid in space is too small! Using smallest grid."
                )
                sz = (max(sz[0], NN), max(sz[1], NN))

        # Convert to polar coordinates
        Pos_radius = np.sqrt(Pos_index_1**2 + Pos_index_2**2)
        pos = Pos_radius > 0
        Pos_angle = np.zeros_like(Pos_radius)
        Pos_angle[pos] = np.arccos(Pos_index_1[pos] / Pos_radius[pos])
        pos = Pos_index_2 < 0
        Pos_angle[pos] = 2 * np.pi - Pos_angle[pos]

        # Extract non-zero entries (for efficiency)
        pos = np.abs(X) > 0  # Could set a threshold here
        Pos_index_1_nz = Pos_index_1[pos]
        Pos_index_2_nz = Pos_index_2[pos]
        Pos_radius_nz = Pos_radius[pos]
        Pos_angle_nz = Pos_angle[pos]
        XX = X[pos]
        XX1 = X1[pos]
        XX2 = X2[pos]

        # Create levels (frequency subbands)
        level = []
        temp_radius = Pos_radius_nz.copy()

        for cnt in range(nbscales):
            level_data = {
                "polar_r": np.array([]),
                "polar_a": np.array([]),
                "index_1": np.array([]),
                "index_2": np.array([]),
                "X_val": np.array([]),
                "X1_val": np.array([]),
                "X2_val": np.array([]),
                "lowpass": np.array([]),
                "hipass": np.array([]),
            }

            pos = temp_radius <= wedge_end[cnt]
            level_data["polar_r"] = Pos_radius_nz[pos]
            level_data["polar_a"] = Pos_angle_nz[pos]
            level_data["index_1"] = Pos_index_1_nz[pos]
            level_data["index_2"] = Pos_index_2_nz[pos]
            level_data["X_val"] = XX[pos]
            level_data["X1_val"] = XX1[pos]
            level_data["X2_val"] = XX2[pos]

            if cnt > 0:
                if not is_cos:
                    # Use fdct_wrapping_window (would need to be implemented)
                    level_data["lowpass"] = fdct_wrapping_window(
                        (wedge_end[cnt] - level_data["polar_r"]) / wedge_length[cnt]
                    )
                else:
                    # Use gdct_cos (would need to be implemented)
                    level_data["lowpass"] = gdct_cos(
                        (wedge_end[cnt] - level_data["polar_r"]) / wedge_length[cnt]
                    )
                level_data["hipass"] = np.sqrt(1 - level_data["lowpass"] ** 2)

            temp_radius[pos] = 1e10  # Mark as processed
            level.append(level_data)

        # Create tiling structure
        tiling = [[] for _ in range(nbscales - 1)]

        for cnt in range(2, nbscales):
            tiling[cnt - 1] = [[[] for _ in range(nbangles[cnt])] for _ in range(2)]

            for cnt2 in range(nbangles[cnt]):
                for cnt3 in range(red):
                    # Process current wedge
                    if cnt2 < nbangles[cnt] - 1:
                        # Standard wedge case
                        pos = (
                            level[cnt]["polar_a"] >= (cnt2 + cnt3 / red) * angles[cnt]
                        ) & (
                            level[cnt]["polar_a"]
                            < (cnt2 + 1 + cnt3 / red) * angles[cnt]
                        )

                        wedge_data = {
                            "polar_a": level[cnt]["polar_a"][pos],
                            "index_1": level[cnt]["index_1"][pos],
                            "index_2": level[cnt]["index_2"][pos],
                            "X_val": level[cnt]["X_val"][pos],
                            "X1_val": level[cnt]["X1_val"][pos],
                            "X2_val": level[cnt]["X2_val"][pos],
                        }

                        if not is_cos:
                            wedge_data["lowpass"] = fdct_wrapping_window(
                                (
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt]
                                )
                                / angles[cnt]
                            )
                        else:
                            wedge_data["lowpass"] = gdct_cos(
                                (
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt]
                                )
                                / angles[cnt]
                            )

                        wedge_data["hipass"] = np.sqrt(1 - wedge_data["lowpass"] ** 2)
                        wedge_data["lowpass"] *= level[cnt]["lowpass"][pos]
                        wedge_data["hipass"] *= level[cnt]["lowpass"][pos]

                        tiling[cnt - 1][0][cnt2].append(wedge_data)

                        # Process next level
                        pos = (
                            level[cnt - 1]["polar_a"]
                            >= (cnt2 + cnt3 / red) * angles[cnt]
                        ) & (
                            level[cnt - 1]["polar_a"]
                            < (cnt2 + 1 + cnt3 / red) * angles[cnt]
                        )

                        wedge_data = {
                            "polar_a": level[cnt - 1]["polar_a"][pos],
                            "index_1": level[cnt - 1]["index_1"][pos],
                            "index_2": level[cnt - 1]["index_2"][pos],
                            "X_val": level[cnt - 1]["X_val"][pos],
                            "X1_val": level[cnt - 1]["X1_val"][pos],
                            "X2_val": level[cnt - 1]["X2_val"][pos],
                        }

                        if not is_cos:
                            wedge_data["lowpass"] = fdct_wrapping_window(
                                (
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt]
                                )
                                / angles[cnt]
                            )
                        else:
                            wedge_data["lowpass"] = gdct_cos(
                                (
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt]
                                )
                                / angles[cnt]
                            )

                        wedge_data["hipass"] = np.sqrt(1 - wedge_data["lowpass"] ** 2)
                        wedge_data["lowpass"] *= level[cnt - 1]["hipass"][pos]
                        wedge_data["hipass"] *= level[cnt - 1]["hipass"][pos]

                        tiling[cnt - 1][1][cnt2].append(wedge_data)
                    else:
                        # Handle wrap-around for last wedge
                        pos = (
                            level[cnt]["polar_a"] >= (cnt2 + cnt3 / red) * angles[cnt]
                        ) | (
                            level[cnt]["polar_a"]
                            < (cnt2 + 1 + cnt3 / red) * angles[cnt] - 2 * np.pi
                        )

                        wedge_data = {
                            "polar_a": level[cnt]["polar_a"][pos],
                            "index_1": level[cnt]["index_1"][pos],
                            "index_2": level[cnt]["index_2"][pos],
                            "X_val": level[cnt]["X_val"][pos],
                            "X1_val": level[cnt]["X1_val"][pos],
                            "X2_val": level[cnt]["X2_val"][pos],
                        }

                        if not is_cos:
                            wedge_data["lowpass"] = fdct_wrapping_window(
                                np.mod(
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt],
                                    2 * np.pi,
                                )
                                / angles[cnt]
                            )
                        else:
                            wedge_data["lowpass"] = gdct_cos(
                                np.mod(
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt],
                                    2 * np.pi,
                                )
                                / angles[cnt]
                            )

                        wedge_data["hipass"] = np.sqrt(1 - wedge_data["lowpass"] ** 2)
                        wedge_data["lowpass"] *= level[cnt]["lowpass"][pos]
                        wedge_data["hipass"] *= level[cnt]["lowpass"][pos]

                        tiling[cnt - 1][0][cnt2].append(wedge_data)

                        # Process next level
                        pos = (
                            level[cnt - 1]["polar_a"]
                            >= (cnt2 + cnt3 / red) * angles[cnt]
                        ) | (
                            level[cnt - 1]["polar_a"]
                            < (cnt2 + 1 + cnt3 / red) * angles[cnt] - 2 * np.pi
                        )

                        wedge_data = {
                            "polar_a": level[cnt - 1]["polar_a"][pos],
                            "index_1": level[cnt - 1]["index_1"][pos],
                            "index_2": level[cnt - 1]["index_2"][pos],
                            "X_val": level[cnt - 1]["X_val"][pos],
                            "X1_val": level[cnt - 1]["X1_val"][pos],
                            "X2_val": level[cnt - 1]["X2_val"][pos],
                        }

                        if not is_cos:
                            wedge_data["lowpass"] = fdct_wrapping_window(
                                np.mod(
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt],
                                    2 * np.pi,
                                )
                                / angles[cnt]
                            )
                        else:
                            wedge_data["lowpass"] = gdct_cos(
                                np.mod(
                                    wedge_data["polar_a"]
                                    - (cnt2 + cnt3 / red) * angles[cnt],
                                    2 * np.pi,
                                )
                                / angles[cnt]
                            )

                        wedge_data["hipass"] = np.sqrt(1 - wedge_data["lowpass"] ** 2)
                        wedge_data["lowpass"] *= level[cnt - 1]["hipass"][pos]
                        wedge_data["hipass"] *= level[cnt - 1]["hipass"][pos]

                        tiling[cnt - 1][1][cnt2].append(wedge_data)

        # Generate coefficients
        # First scale (low frequency)
        temp = np.zeros(sz, dtype=np.complex128)
        idx1 = np.mod(level[0]["index_1"] + pN1, sz[0]).astype(int)
        idx2 = np.mod(level[0]["index_2"], sz[1]).astype(int)
        temp[idx1, idx2] = level[0]["X_val"]

        idx1 = np.mod(level[1]["index_1"] + pN1, sz[0]).astype(int)
        idx2 = np.mod(level[1]["index_2"], sz[1]).astype(int)
        temp[idx1, idx2] = level[1]["X_val"] * level[1]["lowpass"]

        fac = 1  # sqrt(pi * wedge_end[1]**2)
        C[0].append(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp)))
            * np.sqrt(sz[0] * sz[1])
            / fac
        )

        temp = np.zeros(sz, dtype=np.complex128)
        temp[idx1, idx2] = level[0]["X1_val"]
        temp[idx1, idx2] = level[1]["X1_val"] * level[1]["lowpass"]
        C1[0].append(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp)))
            * np.sqrt(sz[0] * sz[1])
            / fac
        )

        temp = np.zeros(sz, dtype=np.complex128)
        temp[idx1, idx2] = level[0]["X2_val"]
        temp[idx1, idx2] = level[1]["X2_val"] * level[1]["lowpass"]
        C2[0].append(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp)))
            * np.sqrt(sz[0] * sz[1])
            / fac
        )

        # Higher scales
        for cnt in range(2, nbscales):
            C[cnt - 1] = [[] for _ in range(nbangles[cnt])]
            C1[cnt - 1] = [[] for _ in range(nbangles[cnt])]
            C2[cnt - 1] = [[] for _ in range(nbangles[cnt])]

            fac = 1  # sqrt(angles[cnt] * (wedge_end[cnt]**2 - wedge_end[cnt-2]**2) / 4)

            for cnt2 in range(nbangles[cnt]):
                for cnt3 in range(red):
                    # Process main coefficients (C)
                    temp = np.zeros(sz, dtype=np.complex128)

                    # Current wedge
                    idx1 = np.mod(
                        tiling[cnt - 1][0][cnt2][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][0][cnt2][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][0][cnt2][cnt3]["lowpass"]
                        * tiling[cnt - 1][0][cnt2][cnt3]["X_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][1][cnt2][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][1][cnt2][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][1][cnt2][cnt3]["lowpass"]
                        * tiling[cnt - 1][1][cnt2][cnt3]["X_val"]
                    )

                    # Next wedge (or first wedge if at end)
                    next_wedge = 0 if cnt2 == nbangles[cnt] - 1 else cnt2 + 1

                    idx1 = np.mod(
                        tiling[cnt - 1][0][next_wedge][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][0][next_wedge][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][0][next_wedge][cnt3]["hipass"]
                        * tiling[cnt - 1][0][next_wedge][cnt3]["X_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][1][next_wedge][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][1][next_wedge][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][1][next_wedge][cnt3]["hipass"]
                        * tiling[cnt - 1][1][next_wedge][cnt3]["X_val"]
                    )

                    C[cnt - 1][cnt2].append(
                        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp)))
                        * np.sqrt(sz[0] * sz[1])
                        / fac
                    )

                    # Process first derivative coefficients (C1)
                    temp = np.zeros(sz, dtype=np.complex128)

                    idx1 = np.mod(
                        tiling[cnt - 1][0][cnt2][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][0][cnt2][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][0][cnt2][cnt3]["lowpass"]
                        * tiling[cnt - 1][0][cnt2][cnt3]["X1_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][1][cnt2][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][1][cnt2][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][1][cnt2][cnt3]["lowpass"]
                        * tiling[cnt - 1][1][cnt2][cnt3]["X1_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][0][next_wedge][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][0][next_wedge][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][0][next_wedge][cnt3]["hipass"]
                        * tiling[cnt - 1][0][next_wedge][cnt3]["X1_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][1][next_wedge][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][1][next_wedge][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][1][next_wedge][cnt3]["hipass"]
                        * tiling[cnt - 1][1][next_wedge][cnt3]["X1_val"]
                    )

                    C1[cnt - 1][cnt2].append(
                        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp)))
                        * np.sqrt(sz[0] * sz[1])
                        / fac
                    )

                    # Process second derivative coefficients (C2)
                    temp = np.zeros(sz, dtype=np.complex128)

                    idx1 = np.mod(
                        tiling[cnt - 1][0][cnt2][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][0][cnt2][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][0][cnt2][cnt3]["lowpass"]
                        * tiling[cnt - 1][0][cnt2][cnt3]["X2_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][1][cnt2][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][1][cnt2][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][1][cnt2][cnt3]["lowpass"]
                        * tiling[cnt - 1][1][cnt2][cnt3]["X2_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][0][next_wedge][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][0][next_wedge][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][0][next_wedge][cnt3]["hipass"]
                        * tiling[cnt - 1][0][next_wedge][cnt3]["X2_val"]
                    )

                    idx1 = np.mod(
                        tiling[cnt - 1][1][next_wedge][cnt3]["index_1"] + pN1, sz[0]
                    ).astype(int)
                    idx2 = np.mod(
                        tiling[cnt - 1][1][next_wedge][cnt3]["index_2"] + pN2, sz[1]
                    ).astype(int)
                    temp[idx1, idx2] = (
                        tiling[cnt - 1][1][next_wedge][cnt3]["hipass"]
                        * tiling[cnt - 1][1][next_wedge][cnt3]["X2_val"]
                    )

                    C2[cnt - 1][cnt2].append(
                        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp)))
                        * np.sqrt(sz[0] * sz[1])
                        / fac
                    )
    else:
        # Case when R_low > 0 (not fully implemented here)
        raise NotImplementedError(
            "Case when R_low > 0 is not fully implemented in this conversion"
        )

    return C, C1, C2


# Note: The fdct_wrapping_window and gdct_cos functions would need to be implemented separately
# as shown in previous conversions
