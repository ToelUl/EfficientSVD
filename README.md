# EfficientSVD
This Python class computes SVD (A = U S Vh) for various matrix types (NumPy, PyTorch, SciPy sparse). It automatically selects the optimal backend (PyTorch, SciPy, or Scikit-learn) based on matrix properties, desired SVD type (full, truncated, randomized), and available libraries, while allowing manual method selection.
