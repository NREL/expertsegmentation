DEFAULT_LAMBDA = 1
SUPPORTED_OBJECTIVES = {"volume_fraction", "connectivity"}
DEFAULT_FILTERS = {
    "gaussian_smoothing": [0.3, 0.7, 1, 1.6, 3.5, 5, 10],
    "laplacian_of_gaussian": [0.7, 1, 1.6, 3.5, 5, 10],
    "gaussian_gradient_magnitude": [0.7, 1, 1.6, 3.5, 5, 10],
    "diff_of_gaussians": [0.7, 1, 1.6, 3.5, 5, 10],
    "structure_tensor_eigvals": [0.7, 1, 1.6, 3.5, 5, 10],
   "hessian_of_gaussian_eigvals": [0.7, 1, 1.6, 3.5, 5, 10]
}