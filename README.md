# EfficientSVD User Documentation

## Interactive Tutorial with Google Colab

Explore the features of `EfficientSVD` interactively using our tutorial notebook hosted on Google Colab. This notebook covers installation, various SVD methods (`auto`, `full`, `truncated`, `randomized`, `values_only`), and usage with different matrix types (NumPy, SciPy sparse, PyTorch tensors).

Click the badge below to launch the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ToelUl/EfficientSVD/blob/main/efficient_svd_tutorial.ipynb)

## Overview

`EfficientSVD` is a Python class providing a unified and efficient interface for computing the Singular Value Decomposition (SVD: A = U S Vh) of various matrix formats, including NumPy arrays, PyTorch tensors, and SciPy sparse matrices. It intelligently selects the optimal backend implementation (from PyTorch, SciPy, Scikit-learn) based on matrix properties, desired computation type (full, truncated, randomized), and available libraries, while also allowing manual method specification.

## Key Features

* **Unified Interface**: A single `compute` method handles diverse input matrix types and SVD computation requirements.
* **Multi-Backend Support**: Leverages efficient SVD implementations from NumPy, PyTorch (with GPU support), SciPy, and Scikit-learn.
* **Automatic Method Selection (`'auto'`)**: Intelligently chooses the most suitable SVD method (`'full'`, `'truncated'`, `'randomized'`, `'values_only'`) based on input characteristics (sparsity, size), desired number of components (`k`), whether singular vectors are needed (`compute_uv`), and library availability.
* **Flexible Configuration**: Allows setting default behaviors during initialization, which can be overridden per computation via the `compute` method.
* **Consistent Output**: Returns results (U, S, Vh or S) as NumPy arrays, regardless of the input type or backend used.

## Dependencies

* **Required**: NumPy (`pip install numpy`)
* **Optional (for full functionality)**:
    * PyTorch (`pip install torch` or follow official instructions for CUDA version): Enables efficient `'full'` and `'values_only'` methods, including GPU acceleration.
    * SciPy (`pip install scipy`): Provides the `'truncated'` SVD implementation (`svds`), particularly effective for sparse matrices.
    * Scikit-learn (`pip install scikit-learn`): Offers `'randomized'` SVD and an alternative `'truncated'` SVD implementation.

Missing optional libraries may restrict available SVD methods or reduce performance.

## API Reference

### Class Initialization

```python
from typing import Optional, Union, Tuple, Literal, Type, Any
import numpy as np
# Assuming type aliases MatrixType, SVDResultType, SVDValsResultType, SVDMethod are defined
# e.g., from universal_svd import MatrixType, ...

class EfficientSVD:
    def __init__(self,
                 method: SVDMethod = 'auto',
                 k: Optional[int] = None,
                 compute_uv: bool = True,
                 random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None):
        """
        Initializes the EfficientSVD configuration.

        Args:
            method: The default SVD computation method ('auto', 'full',
                'truncated', 'randomized', 'values_only').
            k: The default number of singular values/vectors for
                truncated/randomized methods. Must be set if using these methods
                unless overridden in `compute`.
            compute_uv: Default for whether to compute U and Vh vectors.
            random_state: Default seed or generator for randomized algorithms.
        """
        # ... implementation ...
```

**Parameters**:

* `method` (`SVDMethod`, optional, default=`'auto'`): The default SVD computation method. Valid options:
    * `'auto'`: Automatically selects the most suitable backend based on input and configuration.
    * `'full'`: Computes the full SVD. Uses `torch.linalg.svd` (if available) or `np.linalg.svd`. Suitable for dense matrices where all singular values/vectors are needed; can be computationally intensive for large matrices.
    * `'truncated'`: Computes only the largest `k` singular values/vectors. Uses `scipy.sparse.linalg.svds` (if available) or `sklearn.decomposition.TruncatedSVD`. Ideal for sparse matrices or large dense matrices where only top components are required. **Requires `k` to be specified**.
    * `'randomized'`: Computes an approximate truncated SVD for the largest `k` components using randomized algorithms via `sklearn.utils.extmath.randomized_svd`. Often faster than exact truncated methods for large matrices. **Requires `k` to be specified**.
    * `'values_only'`: Computes only the singular values (S). Prefers `torch.linalg.svdvals` if available; otherwise, falls back to computing a full SVD and extracting S.
* `k` (`int`, optional, default=`None`): The default number of singular values/vectors to compute when using `'truncated'` or `'randomized'` methods. Must be provided either at initialization or during the `compute` call if these methods are selected.
* `compute_uv` (`bool`, optional, default=`True`): Specifies whether to compute and return the singular vectors U and Vh by default. If `False`, only the singular values S are returned.
* `random_state` (`int`, `np.random.RandomState`, `np.random.Generator`, optional, default=`None`): The default seed or generator for randomized algorithms (`'randomized'` method or `sklearn.TruncatedSVD` with randomized solver) to ensure reproducibility.

### Primary Method: `compute`

```python
    def compute(self,
                A: MatrixType,
                method: Optional[SVDMethod] = None,
                k: Optional[int] = None,
                compute_uv: Optional[bool] = None,
                random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
                **kwargs) -> Union[SVDResultType, SVDValsResultType]:
        """
        Computes the Singular Value Decomposition (SVD) of matrix A.

        Args:
            A: The input matrix (NumPy array, PyTorch tensor, or SciPy sparse matrix).
            method: The SVD computation method to use for this call, overriding the instance default.
            k: The number of singular values/vectors for truncated/randomized methods for this call,
               overriding the instance default. Required if method is 'truncated' or 'randomized'.
            compute_uv: Whether to compute and return U and Vh vectors for this call,
                        overriding the instance default. If False, only singular values (S)
                        are returned.
            random_state: Seed or generator for randomized algorithms for this call,
                          overriding the instance default.
            **kwargs: Additional keyword arguments passed to the underlying backend function.
                      See documentation of specific backend functions (e.g., `torch.linalg.svd`,
                      `scipy.sparse.linalg.svds`, `sklearn.utils.extmath.randomized_svd`)
                      for available options.

        Returns:
            If `compute_uv` is True: A tuple (U, S, Vh) containing NumPy arrays.
            If `compute_uv` is False: A 1D NumPy array S containing singular values.

        Raises:
            ValueError: If input parameters are invalid.
            TypeError: If the input matrix type is unsupported.
            ImportError: If a required library for the selected method is not installed.
            RuntimeError: If no suitable backend is found or if a backend fails.
            LinAlgError: If the SVD computation itself fails (e.g., convergence error).
        """
        # ... implementation ...
```

**Parameters**:

* `A` (`MatrixType`, required): The input matrix (NumPy `ndarray`, PyTorch `Tensor`, or SciPy `spmatrix`).
* `method` (`SVDMethod`, optional): Overrides the instance's default `method` for this specific call.
* `k` (`int`, optional): Overrides the instance's default `k` for this call. Required if the effective method is `'truncated'` or `'randomized'`.
* `compute_uv` (`bool`, optional): Overrides the instance's default `compute_uv` for this call. Determines the return type.
* `random_state` (optional): Overrides the instance's default `random_state` for this call.
* `**kwargs`: Additional keyword arguments forwarded to the chosen backend SVD function. Common examples include:
    * For `'full'` (`torch.linalg.svd`): `driver` (`'gesvd'`, `'gesvda'`).
    * For `'truncated'` (`scipy.svds`): `tol` (convergence tolerance).
    * For `'truncated'` (`sklearn.TruncatedSVD`): `algorithm` (`'arpack'`, `'randomized'`), `n_iter`, `tol`.
    * For `'randomized'` (`sklearn.randomized_svd`): `n_oversamples`, `n_iter`, `power_iteration_normalizer`, `flip_sign`.

**Returns**:

* If `compute_uv` evaluates to `True`:
    Returns `SVDResultType`, a tuple `(U, S, Vh)` where:
    * `U` (`np.ndarray`): Unitary matrix of shape (M, K) containing left singular vectors. K is `min(M, N)` for `'full'` or `k` for others.
    * `S` (`np.ndarray`): 1D array of shape (K,) containing singular values, sorted descending.
    * `Vh` (`np.ndarray`): Unitary matrix of shape (K, N) containing the conjugate transpose of right singular vectors.
* If `compute_uv` evaluates to `False`:
    Returns `SVDValsResultType`, which is:
    * `S` (`np.ndarray`): 1D array of shape (K,) containing singular values, sorted descending.

**Note**: All returned arrays are NumPy `ndarray` objects for consistency.

**Exceptions**:

* `ValueError`: Invalid input parameters (e.g., non-positive `k`, missing `k` where required).
* `TypeError`: Unsupported input matrix type for `A`.
* `ImportError`: A necessary backend library (PyTorch, SciPy, Scikit-learn) is not installed for the selected method.
* `RuntimeError`: No suitable backend could be found, or an internal error occurred during backend execution.
* `LinAlgError` (from NumPy/SciPy): The underlying SVD algorithm failed to compute the decomposition (e.g., did not converge).

## Usage Examples

```python
import numpy as np
# Assume EfficientSVD class is defined or imported
# from universal_svd import EfficientSVD

# --- Prepare Data ---
M, N = 200, 100
K = 10
A_dense = np.random.rand(M, N)

_SCIPY_AVAILABLE = False
_TORCH_AVAILABLE = False
_SKLEARN_AVAILABLE = False
A_sparse = None
A_torch = None

try:
    from scipy.sparse import random as sparse_random, spmatrix
    A_sparse = sparse_random(M, N, density=0.1, format='csr')
    _SCIPY_AVAILABLE = True
except ImportError:
    print("SciPy not found, skipping sparse example.")

try:
    import torch
    # Ensure float32 for potential GPU efficiency and compatibility
    A_torch = torch.from_numpy(A_dense.astype(np.float32))
    # if torch.cuda.is_available():
    #     A_torch = A_torch.cuda() # Optional: Move to GPU
    _TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not found, skipping PyTorch example.")

try:
    from sklearn.utils.extmath import randomized_svd
    _SKLEARN_AVAILABLE = True
except ImportError:
    print("Scikit-learn not found, skipping randomized SVD example.")


# --- Initialize SVD Computer ---
svd_computer = EfficientSVD(random_state=42) # Set default random state for reproducibility

# --- Example 1: Auto method, compute U, S, Vh, top 10 components ---
print("\n--- Example 1: Auto SVD (k=10) ---")
try:
    # compute_uv=True is default
    U, S, Vh = svd_computer.compute(A_dense, k=K, method='auto')
    print(f"Computed SVD with k={K}. Output shapes: U={U.shape}, S={S.shape}, Vh={Vh.shape}")
    print(f"Top 5 singular values: {S[:5]}")
except Exception as e:
    print(f"Error: {e}")

# --- Example 2: Full SVD ---
print("\n--- Example 2: Full SVD ---")
try:
    # Using a smaller matrix to avoid excessive time/memory consumption
    A_small = A_dense[:50, :30]
    U_f, S_f, Vh_f = svd_computer.compute(A_small, method='full')
    print(f"Computed Full SVD. Output shapes: U={U_f.shape}, S={S_f.shape}, Vh={Vh_f.shape}")
    print(f"Top 5 singular values: {S_f[:5]}")
except Exception as e:
    print(f"Error: {e}")

# --- Example 3: Compute Singular Values Only ---
print("\n--- Example 3: Values Only ---")
try:
    S_only = svd_computer.compute(A_dense, method='values_only', compute_uv=False)
    print(f"Computed Singular Values Only. Shape: {S_only.shape}")
    print(f"Top 5 singular values: {S_only[:5]}")
except Exception as e:
    print(f"Error: {e}")

# --- Example 4: Explicitly use Randomized SVD (requires Scikit-learn) ---
if _SKLEARN_AVAILABLE:
    print("\n--- Example 4: Randomized SVD (k=10) ---")
    try:
        # Pass additional kwargs to the backend (sklearn.utils.extmath.randomized_svd)
        Ur, Sr, Vhr = svd_computer.compute(A_dense, method='randomized', k=K, n_iter=5, n_oversamples=15)
        print(f"Computed Randomized SVD with k={K}. Output shapes: U={Ur.shape}, S={Sr.shape}, Vh={Vhr.shape}")
        print(f"Top 5 singular values: {Sr[:5]}")
    except Exception as e:
        print(f"Error: {e}")

# --- Example 5: Process Sparse Matrix (requires SciPy) ---
if _SCIPY_AVAILABLE and A_sparse is not None:
    print("\n--- Example 5: Truncated SVD on Sparse Matrix (k=10) ---")
    try:
        # 'auto' or 'truncated' are typically suitable for sparse matrices
        Us, Ss, Vhs = svd_computer.compute(A_sparse, method='truncated', k=K)
        print(f"Computed Truncated SVD on Sparse Matrix with k={K}. Output shapes: U={Us.shape}, S={Ss.shape}, Vh={Vhs.shape}")
        print(f"Top 5 singular values: {Ss[:5]}")
    except Exception as e:
        print(f"Error: {e}")

# --- Example 6: Use PyTorch Tensor Input (requires PyTorch) ---
if _TORCH_AVAILABLE and A_torch is not None:
    print("\n--- Example 6: PyTorch Tensor Input (auto, k=10) ---")
    try:
        # Output is consistently NumPy arrays, even with Tensor input
        Upt, Spt, Vhpt = svd_computer.compute(A_torch, k=K, method='auto')
        print(f"Computed SVD from PyTorch Tensor with k={K}. Output shapes: U={Upt.shape}, S={Spt.shape}, Vh={Vhpt.shape}")
        print(f"Output types are NumPy arrays: U({type(Upt)}), S({type(Spt)}), Vh({type(Vhpt)})")
        print(f"Top 5 singular values: {Spt[:5]}")
    except Exception as e:
        print(f"Error: {e}")

```

---

## `EfficientSVD` Code Logic Explanation

This document provides a detailed explanation of the internal logic and workflow of the `EfficientSVD` Python class. The class is designed to offer a unified interface for computing the Singular Value Decomposition (SVD) of various matrix types by intelligently dispatching the computation to appropriate backend libraries.

### Core Workflow and Logic

1.  **Initialization (`__init__`)**:
    * Upon instantiation of a `EfficientSVD` object, the user can specify default settings for the SVD computation method (`method`), the number of singular values/vectors (`k`) for truncated/randomized methods, whether to compute singular vectors (`compute_uv`), and a random state (`random_state`) for stochastic algorithms.
    * The initializer checks for the availability of optional dependencies (PyTorch, SciPy, Scikit-learn) and issues warnings if a selected default method might rely on an unavailable library.

2.  **Method Invocation (`compute`)**:
    * This is the primary public method for users to perform SVD.
    * It accepts the input matrix `A` and allows overriding the instance's default settings for `method`, `k`, `compute_uv`, and `random_state` for the specific computation.
    * It consolidates the provided arguments with the instance defaults.
    * It calls the internal `_validate_input` method to validate the inputs and determine the most appropriate SVD method and parameters to use (`effective_method`, validated `k`).
    * Subsequently, it invokes the `_dispatch_svd` method, passing the validated parameters and the input matrix, to execute the actual SVD computation using the selected backend.
    * Finally, it formats the result returned by `_dispatch_svd` based on the effective `compute_uv` value, ensuring the return type is either a tuple `(U, S, Vh)` or a NumPy array `S`, consistently returning NumPy arrays regardless of the backend used.

3.  **Input Validation and Method Selection (`_validate_input`)**:
    * **Basic Checks**: Verifies that the input matrix `A` is not `None`, is a supported type (NumPy `ndarray`, PyTorch `Tensor`, SciPy `spmatrix`), and is 2-dimensional.
    * **Parameter `k` Validation**: If `k` is provided, it checks if `k` is a positive integer and does not exceed the smaller dimension of the matrix (`min(m, n)`). If it exceeds, a warning is issued, and `k` is adjusted to `min(m, n)`.
    * **Automatic Method Selection (`method == 'auto'`) (Core Logic)**: This logic determines the most efficient SVD method based on matrix properties and user requirements:
        * **If `compute_uv` is `False` (Singular Values only)**:
            * Prefers `'values_only'` (using `torch.linalg.svdvals`) if PyTorch is available and the matrix is a dense type (Tensor or NumPy array).
            * Otherwise (e.g., sparse matrix or no PyTorch), SVD must be computed first to extract `S`. In this scenario:
                * If `k` is specified and relatively small (heuristic: `k < min(m, n) // 2`): Prioritizes `'randomized'` (sklearn), then `'truncated'` (scipy), falling back to `'full'` (numpy/torch).
                * If the matrix is sparse and `k` is not specified: Sets `k = min(m, n) - 1` (required by `scipy.svds`), then chooses `'truncated'` (scipy) or `'randomized'` (sklearn). Raises an error if neither is available.
                * If the matrix is dense, and `k` is unspecified or large: Defaults to `'values_only'`, anticipating a fallback to `'full'` if PyTorch is unavailable.
        * **If `compute_uv` is `True` (U, S, Vh required)**:
            * If `k` is specified:
                * Small `k`: Prioritizes `'randomized'` (sklearn), then `'truncated'` (scipy), falling back to `'full'` (numpy/torch), issuing a warning if falling back.
                * Large `k`: Chooses `'full'`.
            * If the matrix is sparse and `k` is not specified: Issues a warning, sets `k = min(m, n) - 1`, chooses `'truncated'` (scipy) or `'randomized'` (sklearn). Raises an error if neither is available.
            * If the matrix is dense and `k` is not specified: Chooses `'full'`.
    * **Feasibility Checks and Fallbacks**: After auto-selection, verifies if the chosen method is viable given the available libraries. If not, applies fallbacks (e.g., `'randomized'` -> `'truncated'` or `'full'` if scikit-learn is missing).
    * **Final `k` Validation**: Ensures `k` is appropriately set for methods requiring it (`'truncated'`, `'randomized'`). Specifically adjusts `k` if `method` is `'truncated'` using `scipy.svds` and `k >= min(m, n)`, as `svds` requires `k < min(m, n)`.
    * **Library Availability Check**: Performs a final check that the necessary library for the `effective_method` is installed; raises `RuntimeError` otherwise. Issues warnings for potentially inefficient choices (e.g., `'full'` SVD on a sparse matrix).
    * **Return**: Returns the validated/adjusted `A`, `k`, the determined `effective_method`, and `compute_uv`.

4.  **SVD Computation Dispatch (`_dispatch_svd`)**:
    * **Matrix Conversion**: Based on the `effective_method` and available backends, converts the input matrix `A` into the format required by the target backend function (`A_np` for NumPy/SciPy/Sklearn, `A_torch` for PyTorch, `A_scipy` for SciPy/Sklearn sparse methods). This may involve conversions like NumPy to PyTorch Tensor or sparse to dense, potentially incurring overhead. GPU placement is maintained if the input was a CUDA Tensor and PyTorch is used.
    * **Backend Invocation**:
        * `method == 'full'`: Prefers `torch.linalg.svd` if available (leveraging GPU potential), otherwise uses `np.linalg.svd`. Handles the difference that PyTorch returns `V`, requiring computation of `Vh`. Results are converted back to NumPy if the original input was not a Tensor.
        * `method == 'values_only'`: Prefers `torch.linalg.svdvals` if available. Otherwise, recursively calls `_dispatch_svd` with `method='full'` and `compute_uv=False`, extracting only the singular values `S`.
        * `method == 'truncated'`: Prefers `scipy.sparse.linalg.svds` (requires `k < min(m,n)`). Handles the ascending order of singular values returned by `svds` (requires reversal). Manages `compute_uv`. If SciPy is unavailable but Scikit-learn is, falls back to `sklearn.decomposition.TruncatedSVD`. If `U` is required (`compute_uv=True`), issues a warning and redirects to `'randomized'` as `TruncatedSVD` doesn't directly return `U`.
        * `method == 'randomized'`: Uses `sklearn.utils.extmath.randomized_svd`. Ensures the input is in NumPy or SciPy sparse format.
    * **Error Handling**: Employs a `try...except` block to catch potential exceptions from backend libraries (e.g., convergence failures) and provides informative error messages.
    * **Return**: Returns the computed result, either as a tuple `(U, S, Vh)` or a 1D array `S`, ensuring all returned arrays are NumPy `ndarray` objects.

### Summary

The `EfficientSVD` class encapsulates the complexity of various SVD implementations behind a straightforward `compute` interface. Its core strength lies in the `_validate_input` logic, which intelligently selects an optimal or viable SVD strategy based on matrix characteristics, computation requirements, and library availability. The `_dispatch_svd` method then executes the computation using the chosen backend, handling necessary data conversions and ensuring consistent output formatting.

---