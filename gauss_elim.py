#!/usr/bin/env python3
"""Gaussian elimination — solve systems of linear equations."""
import sys

def gauss(A, b):
    n = len(b)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    # Forward elimination
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-12: return None
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]
    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i+1, n))) / M[i][i]
    return x

def determinant(A):
    n = len(A); M = [row[:] for row in A]; det = 1
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if col != max_row: M[col], M[max_row] = M[max_row], M[col]; det *= -1
        if abs(M[col][col]) < 1e-12: return 0
        det *= M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1 if len(M[0]) > n else n):
                M[row][j] -= factor * M[col][j]
    return det

if __name__ == "__main__":
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    x = gauss(A, b)
    print("System: Ax = b")
    for i, row in enumerate(A):
        print(f"  {row} · x = {b[i]}")
    print(f"\nSolution: x = {[round(xi, 6) for xi in x]}")
    # Verify
    for i in range(len(b)):
        calc = sum(A[i][j] * x[j] for j in range(len(x)))
        print(f"  Check row {i}: {calc:.6f} = {b[i]} {'✓' if abs(calc-b[i])<1e-9 else '✗'}")
    print(f"\ndet(A) = {determinant(A):.6f}")
