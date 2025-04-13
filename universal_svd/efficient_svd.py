# -*- coding: utf-8 -*-
"""
Module for computing Singular Value Decomposition (SVD) using various backends.

This module provides the EfficientSVD class, which acts as a unified interface
to different SVD implementations from PyTorch, SciPy, and Scikit-learn,
allowing efficient computation for various matrix types and sizes.
"""

import warnings
from typing import Optional, Union, Tuple, Literal, Type, Any

# Try importing necessary libraries and alias them
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    raise ImportError("NumPy is required for EfficientSVD. Please install it.")

try:
    from scipy.sparse import spmatrix as scipy_sparse_matrix
    from scipy.sparse.linalg import svds as scipy_svds
    _SCIPY_AVAILABLE = True
except ImportError:
    scipy_sparse_matrix = None # type: ignore
    scipy_svds = None # type: ignore
    _SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD
    from sklearn.utils.extmath import randomized_svd as sklearn_randomized_svd
    from sklearn.utils.validation import check_random_state
    _SKLEARN_AVAILABLE = True
except ImportError:
    SklearnTruncatedSVD = None # type: ignore
    sklearn_randomized_svd = None # type: ignore
    check_random_state = None # type: ignore
    _SKLEARN_AVAILABLE = False

# Define supported matrix types
MatrixType = Union[np.ndarray, 'torch.Tensor' if _TORCH_AVAILABLE else np.ndarray, 'scipy_sparse_matrix' if _SCIPY_AVAILABLE else np.ndarray]
SVDResultType = Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]] # U, S, Vh
SVDValsResultType = np.ndarray # S only

# Define backend method literals
SVDMethod = Literal['auto', 'full', 'truncated', 'randomized', 'values_only']

class EfficientSVD:
    """
    Computes Singular Value Decomposition (SVD) using optimal backends.

    This class provides a unified interface to compute SVD (A = U S Vh) for
    various input matrix types (NumPy arrays, PyTorch tensors, SciPy sparse
    matrices) and sizes. It automatically selects or allows manual selection
    of an efficient backend based on the matrix properties, desired computation
    type (full, truncated, randomized), and available libraries.

    Supported Backends:
        - 'full': `torch.linalg.svd` (if available, prefers GPU if tensor is on GPU)
                  or `np.linalg.svd`. Best for dense matrices where all singular
                  values/vectors are needed. Can be slow/memory-intensive for large matrices.
        - 'truncated': `scipy.sparse.linalg.svds` (if available) or
                       `sklearn.decomposition.TruncatedSVD` (if available).
                       Computes only the largest `k` singular values/vectors.
                       Suitable for sparse matrices or large dense matrices where
                       only the top components are required. Requires `k`.
        - 'randomized': `sklearn.utils.extmath.randomized_svd` (if available).
                        Computes an approximate truncated SVD using randomized
                        algorithms. Often faster than exact truncated methods for
                        large dense or sparse matrices. Requires `k`.
        - 'values_only': `torch.linalg.svdvals` (if available) or computes SVD
                         using another method and returns only singular values.
                         Faster if U and Vh are not needed.
        - 'auto': Automatically selects a backend based on matrix type (sparse/dense),
                  size, whether `k` is specified, and `compute_uv`. Prioritizes
                  efficiency (e.g., randomized/truncated for large `k`, sparse methods
                  for sparse matrices, PyTorch for GPU tensors).

    Attributes:
        method (SVDMethod): The default SVD method to use if not specified in `compute`.
        k (Optional[int]): The default number of singular values/vectors to compute
                           for truncated/randomized methods.
        compute_uv (bool): Whether to compute and return U and Vh singular vectors
                           by default.
        random_state (Optional[Union[int, np.random.RandomState]]): Default random
                           state for randomized algorithms.
    """

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
        self.method = method
        self.k = k
        self.compute_uv = compute_uv
        self.random_state = random_state

        # Check for library availability based on potential usage
        if method in ['truncated', 'randomized', 'auto'] and not (_SKLEARN_AVAILABLE or _SCIPY_AVAILABLE):
             warnings.warn("SciPy or Scikit-learn not found. 'truncated' and 'randomized' "
                           "methods may not be available.", ImportWarning)
        if method in ['full', 'values_only', 'auto'] and not _TORCH_AVAILABLE:
            warnings.warn("PyTorch not found. GPU acceleration and some 'full'/'values_only' "
                          "optimizations are unavailable.", ImportWarning)


    def _validate_input(self, A: MatrixType, k: Optional[int], method: SVDMethod, compute_uv: bool) -> Tuple[MatrixType, int, SVDMethod, bool]:
        """Validates input matrix and parameters."""
        if A is None:
            raise ValueError("Input matrix A cannot be None.")

        if not isinstance(A, (np.ndarray,
                             (torch.Tensor if _TORCH_AVAILABLE else type(None)),
                             (scipy_sparse_matrix if _SCIPY_AVAILABLE else type(None))
                            )):
            raise TypeError(f"Unsupported matrix type: {type(A)}. Expected NumPy array, "
                            "PyTorch tensor, or SciPy sparse matrix.")

        if A.ndim != 2:
            raise ValueError(f"Input matrix must be 2-dimensional, got {A.ndim} dimensions.")

        m, n = A.shape
        if k is not None:
            if not isinstance(k, int) or k <= 0:
                raise ValueError(f"k must be a positive integer, got {k}.")
            if k > min(m, n):
                 warnings.warn(f"k={k} is greater than min(m, n)={min(m,n)}. "
                               f"Setting k to {min(m, n)} for truncated/randomized SVD.")
                 k = min(m, n)

        # Check method requirements
        is_sparse = _SCIPY_AVAILABLE and isinstance(A, scipy_sparse_matrix)
        is_tensor = _TORCH_AVAILABLE and isinstance(A, torch.Tensor)

        effective_method = method

        # --- Auto Method Selection Logic ---
        if method == 'auto':
            if not compute_uv:
                # Prefer specialized svdvals if available and matrix is suitable
                if _TORCH_AVAILABLE and (is_tensor or not is_sparse):
                    effective_method = 'values_only'
                else: # Fallback: compute SVD and discard U, Vh later
                    if k is not None and k < min(m,n) // 2 : # Heuristic for large matrix needing few components
                         if _SKLEARN_AVAILABLE:
                              effective_method = 'randomized'
                         elif _SCIPY_AVAILABLE:
                              effective_method = 'truncated'
                         else:
                              effective_method = 'full' # Fallback, might be slow
                    elif is_sparse:
                         if k is None: k = min(m, n) -1 # svds requires k < min(m,n)
                         if _SCIPY_AVAILABLE:
                             effective_method = 'truncated' # svds is usually preferred for sparse
                         elif _SKLEARN_AVAILABLE:
                             effective_method = 'randomized' # sklearn truncated can use arpack too
                         else:
                             raise RuntimeError("Cannot perform SVD on sparse matrix without SciPy or Scikit-learn.")
                    else: # Dense matrix, compute_uv=False, k is large or None
                         effective_method = 'values_only' # Prefer torch.svdvals if possible
            elif k is not None: # compute_uv=True, k is specified
                if k < min(m, n) // 2: # Heuristic: k is relatively small
                     if _SKLEARN_AVAILABLE:
                         effective_method = 'randomized' # Often fastest for approx
                     elif _SCIPY_AVAILABLE:
                         effective_method = 'truncated' # Exact truncated
                     else:
                         effective_method = 'full' # Fallback
                         warnings.warn("k specified but SciPy/Scikit-learn not available for truncated/randomized. "
                                       "Falling back to 'full' SVD, which might be inefficient.", RuntimeWarning)
                else: # k is large, might as well compute full SVD if possible
                    effective_method = 'full'
            elif is_sparse: # compute_uv=True, k is None, sparse matrix
                 if _SCIPY_AVAILABLE or _SKLEARN_AVAILABLE:
                     # Full SVD on sparse is usually not intended/feasible. Default to k=min(shape)-1 for truncated.
                     warnings.warn("Attempting SVD on a sparse matrix without specifying 'k'. "
                                   "Defaulting to truncated SVD with k = min(shape)-1. "
                                   "Specify 'method' and 'k' explicitly for control.", UserWarning)
                     k = min(m, n) - 1
                     if k <= 0: raise ValueError("Matrix dimensions too small for truncated SVD (min(m,n) <= 1).")
                     effective_method = 'truncated' if _SCIPY_AVAILABLE else 'randomized' # Prefer svds if available
                 else:
                     raise RuntimeError("Cannot perform SVD on sparse matrix without SciPy or Scikit-learn.")
            else: # compute_uv=True, k is None, dense matrix
                effective_method = 'full' # Default for dense is full SVD

            # Final check if auto-selected method is viable
            if effective_method == 'values_only' and not _TORCH_AVAILABLE:
                # If torch.svdvals not available, need full SVD to get values
                effective_method = 'full'
            if effective_method in ['truncated', 'randomized'] and k is None:
                raise ValueError(f"Method '{effective_method}' requires 'k' to be specified.")
            if effective_method == 'truncated' and not _SCIPY_AVAILABLE and not _SKLEARN_AVAILABLE:
                warnings.warn("SciPy/Scikit-learn not available for 'truncated' SVD. Falling back to 'full'.", RuntimeWarning)
                effective_method = 'full'
            if effective_method == 'randomized' and not _SKLEARN_AVAILABLE:
                warnings.warn("Scikit-learn not available for 'randomized' SVD. Falling back to 'truncated' or 'full'.", RuntimeWarning)
                effective_method = 'truncated' if _SCIPY_AVAILABLE and k is not None else 'full'

        # --- Validate k for specific methods ---
        if effective_method in ['truncated', 'randomized']:
            if k is None:
                raise ValueError(f"Method '{effective_method}' requires 'k' to be specified.")
            if effective_method == 'truncated' and _SCIPY_AVAILABLE and k >= min(m, n):
                # scipy.sparse.linalg.svds requires k < min(m, n)
                warnings.warn(f"scipy.sparse.linalg.svds requires k < min(shape). Adjusting k from {k} to {min(m, n) - 1}.", RuntimeWarning)
                k = min(m, n) - 1
                if k <= 0:
                    raise ValueError(f"Cannot compute truncated SVD with k={k} for shape ({m}, {n}).")

        # --- Validate method availability ---
        if effective_method == 'full' and not _TORCH_AVAILABLE and not hasattr(np.linalg, 'svd'):
            raise RuntimeError("Full SVD requires PyTorch or NumPy.")
        if effective_method == 'values_only' and not _TORCH_AVAILABLE:
            # Will fallback to computing full SVD and extracting S
             warnings.warn("PyTorch not found for optimized 'values_only'. Computing full SVD instead.", RuntimeWarning)
        if effective_method == 'truncated' and not _SCIPY_AVAILABLE and not _SKLEARN_AVAILABLE:
             raise RuntimeError("Truncated SVD requires SciPy or Scikit-learn.")
        if effective_method == 'randomized' and not _SKLEARN_AVAILABLE:
             raise RuntimeError("Randomized SVD requires Scikit-learn.")
        if is_sparse and effective_method == 'full':
             warnings.warn("Computing 'full' SVD on a sparse matrix is generally inefficient "
                           "and may lead to memory issues. Consider 'truncated' or 'randomized'.", UserWarning)

        # Ensure k is int if not None
        k_int = int(k) if k is not None else 0 # Use 0 or some marker? k should be validated now.

        return A, k_int, effective_method, compute_uv


    def _dispatch_svd(self, A: MatrixType, k: int, method: SVDMethod, compute_uv: bool, **kwargs) -> Union[SVDResultType, SVDValsResultType]:
        """Dispatches the SVD computation to the appropriate backend."""

        m, n = A.shape
        is_sparse = _SCIPY_AVAILABLE and isinstance(A, scipy_sparse_matrix)
        is_tensor = _TORCH_AVAILABLE and isinstance(A, torch.Tensor)
        random_state = kwargs.get('random_state', self.random_state)

        # --- Convert matrix if necessary for the chosen backend ---
        # Note: Conversions can incur memory/time cost.
        A_np: Optional[np.ndarray] = None
        A_torch: Optional[torch.Tensor] = None
        A_scipy: Optional[scipy_sparse_matrix] = None # type: ignore

        if method in ['full', 'values_only'] and _TORCH_AVAILABLE:
            # PyTorch prefers tensors
            if is_tensor:
                A_torch = A
            elif is_sparse:
                 warnings.warn("Converting sparse matrix to dense for PyTorch SVD. This may consume significant memory.", RuntimeWarning)
                 A_np = A.toarray()
                 A_torch = torch.from_numpy(A_np)
            else: # A is NumPy array
                 A_torch = torch.from_numpy(A)
            # Move to GPU if original tensor was on GPU
            if is_tensor and A.is_cuda and hasattr(A_torch, 'cuda'):
                 A_torch = A_torch.cuda(A.device) # type: ignore
        elif method in ['truncated', 'randomized'] or (method=='full' and not _TORCH_AVAILABLE):
             # SciPy/Sklearn/NumPy prefer NumPy arrays or SciPy sparse
             if is_sparse:
                 A_scipy = A # Keep as sparse for svds/sklearn
                 # Sklearn randomized_svd handles sparse directly
                 # Sklearn TruncatedSVD handles sparse directly
                 # Scipy svds handles sparse directly
             elif is_tensor:
                 A_np = A.cpu().numpy() # Move tensor to CPU and convert
             else: # A is NumPy array
                 A_np = A
        # --- Perform Computation ---
        try:
            if method == 'full':
                if _TORCH_AVAILABLE and A_torch is not None:
                    # Use PyTorch SVD (potentially GPU accelerated)
                    # Note: torch.linalg.svd(driver='gesvd') might be faster for tall/thin
                    # but 'gesvda' (default) is generally good. Let PyTorch choose.
                    driver = kwargs.get('driver', None) # Allow specifying driver
                    # PyTorch >= 1.8 returns V, not Vh by default. full_matrices=True is default
                    U, S, V = torch.linalg.svd(A_torch, full_matrices=True, driver=driver)
                    Vh = V.mH # Compute conjugate transpose for Vh
                    # Convert back to NumPy if original wasn't tensor
                    if not is_tensor:
                        U_np = U.cpu().numpy()
                        S_np = S.cpu().numpy()
                        Vh_np = Vh.cpu().numpy()
                    else: # Return tensors if input was tensor? Or always NumPy? Let's stick to NumPy for consistency.
                        U_np = U.cpu().numpy()
                        S_np = S.cpu().numpy()
                        Vh_np = Vh.cpu().numpy()

                    return U_np, S_np, Vh_np if compute_uv else S_np

                elif A_np is not None: # Use NumPy SVD
                    U_np, S_np, Vh_np = np.linalg.svd(A_np, full_matrices=True, compute_uv=True)
                    return U_np, S_np, Vh_np if compute_uv else S_np
                else:
                     raise RuntimeError("No suitable backend found for 'full' SVD.") # Should not happen after validation

            elif method == 'values_only':
                 if _TORCH_AVAILABLE and A_torch is not None:
                     S = torch.linalg.svdvals(A_torch)
                     S_np = S.cpu().numpy()
                     return S_np # Always return NumPy array for consistency
                 else: # Fallback: Compute full SVD and extract S
                     # This case uses the logic from method=='full' but only returns S
                     svd_result = self._dispatch_svd(A, k, 'full', False, **kwargs)
                     return svd_result # Returns S_np directly


            elif method == 'truncated':
                 if k is None: raise ValueError("'k' must be provided for truncated SVD.")
                 # Prefer SciPy svds if available, otherwise Sklearn TruncatedSVD
                 if _SCIPY_AVAILABLE and (A_scipy is not None or A_np is not None):
                     # scipy.sparse.linalg.svds
                     # Requires k < min(M, N)
                     if k >= min(m,n): k = min(m,n) -1 # Already warned, ensure here
                     if k <= 0: raise ValueError("k must be > 0 for svds")

                     mat_for_svds = A_scipy if A_scipy is not None else A_np
                     n_iter = kwargs.get('n_iter', 5) # svds doesn't have n_iter, maybe use tol?
                     tol = kwargs.get('tol', 0) # Tolerance for convergence

                     # which='LM' finds k largest singular values
                     U_np, S_np, Vh_np = scipy_svds(mat_for_svds, k=k, which='LM',
                                                     return_singular_vectors=True if compute_uv else "vh", # 'vh' returns only S, Vh
                                                     tol=tol)

                     # svds returns singular values in ascending order. Reverse them.
                     idx = np.argsort(S_np)[::-1]
                     S_np = S_np[idx]
                     if compute_uv:
                         U_np = U_np[:, idx]
                         Vh_np = Vh_np[idx, :]
                         return U_np, S_np, Vh_np
                     else:
                         # If return_singular_vectors was 'vh', U_np is None
                         Vh_np = Vh_np[idx, :] # Need to get Vh if requested
                         # Rerun if U was needed but we only got Vh? No, API should handle compute_uv.
                         # Rerun if only S is needed:
                         if not compute_uv: # Only singular values needed
                              _ , S_np_vals, _ = scipy_svds(mat_for_svds, k=k, which='LM', return_singular_vectors=False, tol=tol)
                              return np.sort(S_np_vals)[::-1] # Sort descending
                         else: # Need U and Vh
                             # We ran with return_singular_vectors=True
                              return U_np, S_np, Vh_np

                 elif _SKLEARN_AVAILABLE and (A_np is not None or A_scipy is not None):
                     # Use sklearn.decomposition.TruncatedSVD
                     mat_for_sklearn = A_scipy if A_scipy is not None else A_np
                     tsvd = SklearnTruncatedSVD(n_components=k,
                                                algorithm=kwargs.get('solver', 'randomized'), # Default is randomized
                                                n_iter=kwargs.get('n_iter', 5),
                                                random_state=random_state,
                                                tol=kwargs.get('tol', 0.))
                     # Fit and get components
                     tsvd.fit(mat_for_sklearn)
                     S_np = tsvd.singular_values_
                     # Sklearn returns components (Vh) and computes U via transform
                     # This is less direct if U is needed. Let's use randomized_svd if possible,
                     # or stick to svds if available. If only sklearn TuncatedSVD is here:
                     if compute_uv:
                         # U = tsvd.transform(A) / S # This reconstructs U * S. Need U.
                         # This requires computing U explicitly if needed, which TruncatedSVD doesn't store directly.
                         # It's better to use randomized_svd or svds if U is needed.
                         # Let's fall back to randomized_svd if U is needed and only sklearn is available.
                         warnings.warn("Sklearn TruncatedSVD is less direct for getting U. "
                                       "Using randomized_svd instead as compute_uv=True.", UserWarning)
                         return self._dispatch_svd(A, k, 'randomized', compute_uv=True, **kwargs)
                     else:
                         # Only S needed, or S and Vh
                         Vh_np = tsvd.components_
                         return None, S_np, Vh_np if compute_uv else S_np # U is None
                 else:
                     raise RuntimeError("No suitable backend found for 'truncated' SVD.")


            elif method == 'randomized':
                if not _SKLEARN_AVAILABLE: raise RuntimeError("Randomized SVD requires Scikit-learn.")
                if k is None: raise ValueError("'k' must be provided for randomized SVD.")

                mat_for_sklearn = A_scipy if A_scipy is not None else A_np
                if mat_for_sklearn is None: # Should only happen if input was tensor and only sklearn available
                    if is_tensor: A_np = A.cpu().numpy()
                    else: raise ValueError("Cannot determine matrix format for randomized_svd")
                    mat_for_sklearn = A_np

                n_oversamples = kwargs.get('n_oversamples', 10)
                n_iter = kwargs.get('n_iter', 'auto') # default in sklearn
                power_iteration_normalizer = kwargs.get('power_iteration_normalizer', 'auto')
                flip_sign = kwargs.get('flip_sign', True)

                U_np, S_np, Vh_np = sklearn_randomized_svd(mat_for_sklearn,
                                                            n_components=k,
                                                            n_oversamples=n_oversamples,
                                                            n_iter=n_iter,
                                                            power_iteration_normalizer=power_iteration_normalizer,
                                                            flip_sign=flip_sign,
                                                            random_state=random_state)
                # Returns U, S, Vh directly
                return U_np, S_np, Vh_np if compute_uv else S_np

            else:
                raise ValueError(f"Internal error: Unknown effective method '{method}'")

        except Exception as e:
            # Catch potential errors from backends (e.g., convergence issues in iterative methods)
            print(f"Error during SVD computation with method '{method}' on backend.")
            print(f"Matrix type: {type(A)}, shape: {A.shape}, k: {k}, compute_uv: {compute_uv}")
            if is_sparse: print(f"NNZ: {A.nnz}") # type: ignore
            if A_torch is not None: print(f"PyTorch Tensor device: {A_torch.device}")
            raise e


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
            method: The SVD computation method ('auto', 'full', 'truncated',
                'randomized', 'values_only'). Overrides the class default.
            k: The number of singular values/vectors for truncated/randomized
                methods. Overrides the class default. Required if method is
                'truncated' or 'randomized'.
            compute_uv: Whether to compute and return U and Vh vectors.
                Overrides the class default. If False, only singular values (S)
                are returned (as a NumPy array).
            random_state: Seed or generator for randomized algorithms. Overrides
                the class default.
            **kwargs: Additional keyword arguments passed to the underlying
                backend function. Common examples include:
                - For 'full' (torch.linalg.svd): `driver` ('gesvd', 'gesvda').
                - For 'truncated' (scipy.svds): `tol` (convergence tolerance).
                - For 'truncated' (sklearn.TruncatedSVD): `algorithm` ('arpack', 'randomized'),
                  `n_iter`, `tol`.
                - For 'randomized' (sklearn.randomized_svd): `n_oversamples`,
                  `n_iter`, `power_iteration_normalizer`, `flip_sign`.

        Returns:
            If `compute_uv` is True: A tuple (U, S, Vh) containing:
                - U (np.ndarray): Unitary matrix having left singular vectors as columns.
                                  Shape (M, K) where K is min(M,N) for 'full' or `k` for others.
                - S (np.ndarray): The singular values, sorted in descending order. Shape (K,).
                - Vh (np.ndarray): Unitary matrix having right singular vectors as rows
                                   (conjugate transpose of V). Shape (K, N).
            If `compute_uv` is False:
                - S (np.ndarray): The singular values, sorted in descending order. Shape (K,).

            Note: All returned arrays are NumPy arrays for consistency, regardless
                  of the input matrix type or backend used.

        Raises:
            ValueError: If input parameters are invalid (e.g., invalid `k`, missing `k`
                for required methods).
            TypeError: If the input matrix type is unsupported.
            ImportError: If a required library (PyTorch, SciPy, Scikit-learn) for
                the selected method is not installed.
            RuntimeError: If no suitable backend can be found or if a backend fails.
            LinAlgError: If the SVD computation fails (e.g., does not converge). From NumPy/SciPy.
        """
        # Use instance defaults if parameters are not provided
        method_to_use = method if method is not None else self.method
        k_to_use = k if k is not None else self.k
        compute_uv_to_use = compute_uv if compute_uv is not None else self.compute_uv
        # Combine random states, overriding class default if provided here
        rs_to_use = random_state if random_state is not None else self.random_state
        if 'random_state' not in kwargs:
             kwargs['random_state'] = rs_to_use # Pass down consolidated random state

        # Validate inputs and determine the effective method and k
        A_val, k_val, effective_method, compute_uv_val = self._validate_input(
            A, k_to_use, method_to_use, compute_uv_to_use
        )

        # Dispatch to the chosen backend
        result = self._dispatch_svd(A_val, k_val, effective_method, compute_uv_val, **kwargs)

        # Ensure correct return type based on compute_uv
        if compute_uv_val:
             # Expecting (U, S, Vh)
             if not (isinstance(result, tuple) and len(result) == 3):
                 raise RuntimeError(f"Internal error: Expected (U, S, Vh) tuple from backend for method '{effective_method}' but got {type(result)}")
             return result # (U, S, Vh) as NumPy arrays
        else:
             # Expecting S only
             if not isinstance(result, np.ndarray) or result.ndim != 1:
                  # Maybe the backend returned (None, S, None)? Extract S.
                  if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], np.ndarray) and result[1].ndim == 1:
                      return result[1]
                  raise RuntimeError(f"Internal error: Expected S (1D NumPy array) from backend for method '{effective_method}' but got {type(result)}")
             return result # S as NumPy array


# Example Usage (Optional - can be run if script is executed directly)
if __name__ == '__main__':
    print("EfficientSVD Example Usage")

    # --- Configuration ---
    M, N = 1000, 500
    K_COMPONENTS = 10
    USE_SPARSE = False # Set to True to test sparse matrix
    SPARSITY = 0.01

    # --- Create Matrix ---
    np.random.seed(42)
    if USE_SPARSE:
        if not _SCIPY_AVAILABLE:
            print("\nSkipping sparse example: SciPy not available.")
        else:
            from scipy.sparse import random as sparse_random
            print(f"\nCreating a sparse matrix ({M}x{N}) with sparsity {SPARSITY}")
            matrix = sparse_random(M, N, density=SPARSITY, format='csr', random_state=42)
    else:
        print(f"\nCreating a dense NumPy matrix ({M}x{N})")
        matrix = np.random.rand(M, N).astype(np.float32) # Use float32 for potential GPU speedup

    # --- Instantiate EfficientSVD ---
    svd_computer = EfficientSVD(random_state=42) # Set default random state

    # --- Example 1: Auto method (likely randomized/truncated for large matrix) ---
    print("\n--- Example 1: Auto SVD (compute_uv=True, k=10) ---")
    try:
        U, S, Vh = svd_computer.compute(matrix, k=K_COMPONENTS, compute_uv=True, method='auto') # type: ignore
        print(f"Method automatically chosen (likely truncated/randomized).")
        print(f"Computed SVD with k={K_COMPONENTS}")
        print(f"U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}")
        print(f"Top 5 Singular Values: {S[:5]}")
        # Reconstruction error (for truncated/randomized)
        if U is not None and S is not None and Vh is not None:
             reconstructed_matrix = U @ np.diag(S) @ Vh
             if isinstance(matrix, np.ndarray):
                 norm_diff = np.linalg.norm(matrix - reconstructed_matrix)
                 norm_orig = np.linalg.norm(matrix)
                 print(f"Relative reconstruction error: {norm_diff / norm_orig:.4e}")
             elif scipy_sparse_matrix and isinstance(matrix, scipy_sparse_matrix):
                 # Approximate norm difference for sparse
                 # Note: Direct subtraction is dense. This is illustrative.
                 # A better check might involve matrix-vector products.
                 print(f"Reconstruction check applicable for dense. Skipping for sparse.")

    except Exception as e:
        print(f"Error during Auto SVD: {e}")


    # --- Example 2: Full SVD (might be slow/memory intensive) ---
    # Reduce matrix size for full SVD example if needed
    matrix_small = matrix[:100, :50] if not USE_SPARSE else matrix
    if isinstance(matrix_small, np.ndarray) or _TORCH_AVAILABLE:
         print("\n--- Example 2: Full SVD (compute_uv=True) ---")
         try:
             U_full, S_full, Vh_full = svd_computer.compute(matrix_small, method='full', compute_uv=True) # type: ignore
             print(f"Computed Full SVD.")
             print(f"U_full shape: {U_full.shape}, S_full shape: {S_full.shape}, Vh_full shape: {Vh_full.shape}")
             print(f"Top 5 Singular Values: {S_full[:5]}")
             # Full reconstruction error (should be near zero for dense)
             if isinstance(matrix_small, np.ndarray):
                 reconstructed_full = U_full[:, :S_full.size] @ np.diag(S_full) @ Vh_full # Adjust Vh slice for non-square
                 norm_diff_full = np.linalg.norm(matrix_small - reconstructed_full)
                 norm_orig_full = np.linalg.norm(matrix_small)
                 print(f"Relative reconstruction error (full): {norm_diff_full / norm_orig_full:.4e}")

         except Exception as e:
             print(f"Error during Full SVD: {e}")
    else:
        print("\nSkipping Full SVD example for sparse matrix without PyTorch conversion warning enabled.")

    # --- Example 3: Values Only (using PyTorch if available) ---
    if _TORCH_AVAILABLE or isinstance(matrix, np.ndarray):
        print("\n--- Example 3: Values Only ---")
        try:
            S_vals_only = svd_computer.compute(matrix, method='values_only', compute_uv=False) # type: ignore
            print(f"Computed Singular Values Only.")
            print(f"S_vals_only shape: {S_vals_only.shape}")
            print(f"Top 5 Singular Values: {S_vals_only[:5]}")
        except Exception as e:
            print(f"Error during Values Only SVD: {e}")

    # --- Example 4: Randomized SVD explicitly ---
    if _SKLEARN_AVAILABLE:
        print("\n--- Example 4: Randomized SVD (compute_uv=True, k=10) ---")
        try:
            U_rand, S_rand, Vh_rand = svd_computer.compute(matrix, method='randomized', k=K_COMPONENTS, compute_uv=True, n_iter=7) # type: ignore
            print(f"Computed Randomized SVD with k={K_COMPONENTS}, n_iter=7.")
            print(f"U_rand shape: {U_rand.shape}, S_rand shape: {S_rand.shape}, Vh_rand shape: {Vh_rand.shape}")
            print(f"Top 5 Singular Values: {S_rand[:5]}")
        except Exception as e:
            print(f"Error during Randomized SVD: {e}")
    else:
        print("\nSkipping Randomized SVD example: Scikit-learn not available.")


    # --- Example 5: Using PyTorch Tensor Input (if available) ---
    if _TORCH_AVAILABLE:
        print("\n--- Example 5: PyTorch Tensor Input (auto, k=10) ---")
        try:
             # Create a dense tensor for this example
             matrix_torch = torch.from_numpy(np.random.rand(200, 100).astype(np.float32))
             # Uncomment to test GPU (if available)
             if torch.cuda.is_available():
                 matrix_torch = matrix_torch.cuda()
                 print(f"Matrix moved to GPU: {matrix_torch.device}")

             U_pt, S_pt, Vh_pt = svd_computer.compute(matrix_torch, k=K_COMPONENTS, compute_uv=True, method='auto') # type: ignore
             print(f"Computed SVD from PyTorch Tensor input (k={K_COMPONENTS}).")
             print(f"Output types: U({type(U_pt)}), S({type(S_pt)}), Vh({type(Vh_pt)})") # Should be numpy
             print(f"U shape: {U_pt.shape}, S shape: {S_pt.shape}, Vh shape: {Vh_pt.shape}")
             print(f"Top 5 Singular Values: {S_pt[:5]}")

        except Exception as e:
             print(f"Error during PyTorch Tensor SVD: {e}")
    else:
        print("\nSkipping PyTorch Tensor example: PyTorch not available.")