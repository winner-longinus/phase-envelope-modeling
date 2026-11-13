"""Peng-Robinson compositional EOS implementation based on Appendix C.

Reference equations reproduced: C-19 through C-39 (Al-Safran & Brill,
Applied Multiphase Flow in Pipes and Flow Assurance, Appendix C).

Units used by this module:
    Pressure: psia
    Temperature: deg R
    Critical pressure: psia
    Critical temperature: deg R
    Molecular weight: lbm/lbmol
    Density: lbm/ft^3
    Gas constant: 10.7316 psia*ft^3/(lbmol*deg R)

This is a computational implementation of the PR EOS/VLE core, not a
claim that every empirical correlation in Appendix C has been implemented.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import brentq

R = 10.7316  # psia ft^3 / (lbmol R)
SQRT2 = np.sqrt(2.0)

@dataclass(frozen=True)
class Component:
    name: str
    Tc: float       # deg R
    Pc: float       # psia
    omega: float    # acentric factor
    Mw: float       # lbm/lbmol
    Tb: float | None = None  # deg R, optional; not needed by PR EOS
    Vc: float | None = None  # ft^3/lbmol, optional; for Appendix C Eq. C-32

    def validate(self):
        if self.Tc <= 0 or self.Pc <= 0 or self.Mw <= 0:
            raise ValueError(f"Invalid positive property for {self.name}")


def _as_composition(values: Sequence[float], n: int, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.shape != (n,) or np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be {n} finite, nonnegative values")
    total = a.sum()
    if total <= 0:
        raise ValueError(f"{name} must have a positive sum")
    return a / total


def pr_m(component: Component) -> float:
    """Appendix C Eq. C-27; use C-28 for omega > 0.49."""
    w = component.omega
    if w > 0.49:
        return 0.3796 + 1.4850*w - 0.1644*w**2 + 0.01667*w**3
    return 0.37464 + 1.54226*w - 0.26992*w**2


def pr_pure_parameters(components: Sequence[Component], T: float):
    """Return ai(T), bi and alpha_i using Appendix C Eqs. C-24 to C-28."""
    if T <= 0:
        raise ValueError("Temperature must be in deg R and greater than zero")
    for c in components:
        c.validate()
    Tc = np.array([c.Tc for c in components])
    Pc = np.array([c.Pc for c in components])
    omega = np.array([c.omega for c in components])
    m = np.array([pr_m(c) for c in components])
    Tr = T / Tc
    alpha = (1.0 + m*(1.0 - np.sqrt(Tr)))**2
    ai = 0.45724 * R**2 * Tc**2 / Pc * alpha
    bi = 0.0778 * R * Tc / Pc
    return ai, bi, alpha, m


def appendix_c_kij(components: Sequence[Component], user_kij=None) -> np.ndarray:
    """Build BIP matrix. Implements Appendix C's C1/C7+ Eq. C-32 and
    Table C-2 nonhydrocarbon/hydrocarbon values when identifiable.

    A supplied matrix takes precedence. For unlisted pairs the Appendix C
    default is zero. C7+ critical volume must be supplied to use Eq. C-32.
    """
    n = len(components)
    if user_kij is not None:
        k = np.asarray(user_kij, dtype=float)
        if k.shape != (n, n) or not np.allclose(k, k.T, atol=1e-12):
            raise ValueError("kij must be a symmetric n x n matrix")
        if not np.allclose(np.diag(k), 0.0, atol=1e-12):
            raise ValueError("kij diagonal must be zero")
        return k.copy()
    k = np.zeros((n, n), dtype=float)
    nonhc = {"N2", "CO2", "H2S"}
    table = {
        "N2": {"CO2": 0.000, "H2S": 0.130},
        "CO2": {"H2S": 0.1350},
        "H2S": {},
    }
    # Table C-2 values are listed with nonhydrocarbon as the column family
    # and hydrocarbon as the row family.
    hydro_table = {
        "C1": {"N2": 0.025, "CO2": 0.1050, "H2S": 0.070},
        "C2": {"N2": 0.010, "CO2": 0.1300, "H2S": 0.085},
        "C3": {"N2": 0.090, "CO2": 0.1250, "H2S": 0.080},
        "i-C4": {"N2": 0.095, "CO2": 0.1200, "H2S": 0.075},
        "n-C4": {"N2": 0.095, "CO2": 0.1150, "H2S": 0.075},
        "i-C5": {"N2": 0.100, "CO2": 0.1150, "H2S": 0.070},
        "n-C5": {"N2": 0.110, "CO2": 0.1150, "H2S": 0.070},
        "C6": {"N2": 0.110, "CO2": 0.1150, "H2S": 0.055},
        "C7+": {"N2": 0.110, "CO2": 0.1150, "H2S": 0.050},
    }
    names = [c.name for c in components]
    for i, ci in enumerate(components):
        for j in range(i):
            cj = components[j]
            val = None
            if ci.name in nonhc and cj.name in nonhc:
                val = table.get(ci.name, {}).get(cj.name)
                if val is None:
                    val = table.get(cj.name, {}).get(ci.name)
            elif ci.name in nonhc and cj.name in hydro_table:
                val = hydro_table[cj.name].get(ci.name)
            elif cj.name in nonhc and ci.name in hydro_table:
                val = hydro_table[ci.name].get(cj.name)
            if val is not None:
                k[i, j] = k[j, i] = val
    # Appendix C Eq. C-32 for methane / C7+ pair.
    if "C1" in names and "C7+" in names:
        i, j = names.index("C1"), names.index("C7+")
        vc1, vc7 = components[i].Vc, components[j].Vc
        if vc1 is not None and vc7 is not None:
            ratio = 2.0 * vc1**(1/6) * vc7**(1/6) / (vc1**(1/3) + vc7**(1/3))
            k[i, j] = k[j, i] = 0.18 * (1.0 - ratio**6)
    return k


def pr_mix_parameters(P: float, T: float, composition: Sequence[float],
                      components: Sequence[Component], phase: str,
                      kij=None):
    """Appendix C Eqs. C-20 to C-26 (vapour) or C-35 to C-38 (liquid)."""
    if P <= 0 or T <= 0:
        raise ValueError("P and T must be positive absolute values")
    n = len(components)
    z = _as_composition(composition, n, "composition")
    k = appendix_c_kij(components, kij)
    ai, bi, _, _ = pr_pure_parameters(components, T)
    A_i = ai * P / (R*T)**2
    B_i = bi * P / (R*T)
    Aij = np.sqrt(np.outer(A_i, A_i)) * (1.0 - k)
    A = float(z @ Aij @ z)
    B = float(z @ B_i)
    return A, B, Aij, B_i, z


def pr_z_roots(A: float, B: float) -> np.ndarray:
    """Solve Appendix C cubic EOS polynomial (C-19/C-34)."""
    coeff = [1.0, -(1.0-B), A-3.0*B**2-2.0*B,
             -(A*B-B**2-B**3)]
    roots = np.roots(coeff)
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    real = np.sort(real[real > B + 1e-12])
    if len(real) == 0:
        raise RuntimeError("No physically admissible real PR Z root")
    return real


def pr_fugacity_coefficients(P: float, T: float, composition: Sequence[float],
                             components: Sequence[Component], phase: str,
                             kij=None) -> tuple[np.ndarray, float]:
    """Component fugacity coefficients for PR EOS (Appendix C C-10 to C-12).

    phase='vapor' selects the largest admissible Z root; phase='liquid'
    selects the smallest. Returns (phi_i, Z).
    """
    A, B, Aij, B_i, z = pr_mix_parameters(P, T, composition, components, phase, kij)
    roots = pr_z_roots(A, B)
    Z = float(roots[-1] if phase.lower().startswith("v") else roots[0])
    if B <= 0 or Z <= B:
        raise RuntimeError("Invalid PR state: require B>0 and Z>B")
    log_arg_num = Z + (1.0 + SQRT2)*B
    log_arg_den = Z + (1.0 - SQRT2)*B
    if log_arg_num <= 0 or log_arg_den <= 0:
        raise RuntimeError("Invalid logarithm argument in PR fugacity expression")
    sum_aij = Aij @ z
    term = 2.0*sum_aij/A - B_i/B if A > 1e-16 else -B_i/B
    lnphi = (B_i/B)*(Z-1.0) - np.log(Z-B)
    if A > 1e-16 and B > 1e-16:
        lnphi -= (A/(2.0*SQRT2*B))*term*np.log(log_arg_num/log_arg_den)
    return np.exp(lnphi), Z


def equilibrium_K(P: float, T: float, x: Sequence[float], y: Sequence[float],
                   components: Sequence[Component], kij=None) -> np.ndarray:
    """K_i = phi_i^L / phi_i^V at thermodynamic VLE (Appendix C Eq. C-10)."""
    phiL, _ = pr_fugacity_coefficients(P, T, x, components, "liquid", kij)
    phiV, _ = pr_fugacity_coefficients(P, T, y, components, "vapor", kij)
    return phiL / phiV


def gas_density(P: float, T: float, y: Sequence[float],
                components: Sequence[Component], kij=None) -> tuple[float, float]:
    """Gas density from Appendix C Eq. C-33. Returns (rho_v, Z_v)."""
    y = _as_composition(y, len(components), "vapor composition")
    _, Z = pr_fugacity_coefficients(P, T, y, components, "vapor", kij)
    Mw_mix = sum(y[i]*components[i].Mw for i in range(len(components)))
    return P*Mw_mix/(Z*R*T), Z


def liquid_density_eos(P: float, T: float, x: Sequence[float],
                       components: Sequence[Component], kij=None) -> tuple[float, float]:
    """Liquid engineering-EOS density from Appendix C Eq. C-39.
    Returns (rho_L, Z_L). Volume translation is not applied.
    """
    x = _as_composition(x, len(components), "liquid composition")
    _, Z = pr_fugacity_coefficients(P, T, x, components, "liquid", kij)
    Mw_mix = sum(x[i]*components[i].Mw for i in range(len(components)))
    return P*Mw_mix/(Z*R*T), Z


def rachford_rice(beta: float, z: np.ndarray, K: np.ndarray) -> float:
    return float(np.sum(z*(K-1.0)/(1.0 + beta*(K-1.0))))


def flash(P: float, T: float, z: Sequence[float], components: Sequence[Component],
          kij=None, tol: float = 1e-9, maxiter: int = 200):
    """Isothermal-isobaric two-phase flash using PR fugacity equality.

    Returns phase='liquid' or 'vapor' for single-phase states, otherwise
    vapor_fraction, x, y, K, ZL, ZV, rhoL and rhoV.
    """
    z = _as_composition(z, len(components), "overall composition")
    Tc = np.array([c.Tc for c in components]); Pc = np.array([c.Pc for c in components])
    omega = np.array([c.omega for c in components])
    # Wilson K values are only initial guesses; fugacity iteration replaces them.
    lnK = np.log(Pc/P) + 5.373*(1.0+omega)*(1.0-Tc/T)
    K = np.exp(np.clip(lnK, -30, 30))
    f0, f1 = rachford_rice(0.0, z, K), rachford_rice(1.0, z, K)
    if f0 < 0:
        return {"phase": "liquid", "x": z, "vapor_fraction": 0.0}
    if f1 > 0:
        return {"phase": "vapor", "y": z, "vapor_fraction": 1.0}
    beta = brentq(rachford_rice, 0.0, 1.0, args=(z, K), xtol=1e-12)
    for iteration in range(maxiter):
        x = z/(1.0 + beta*(K-1.0)); y = K*x
        x /= x.sum(); y /= y.sum()
        K_new = equilibrium_K(P, T, x, y, components, kij)
        lnK_new = np.log(np.maximum(K_new, 1e-30))
        lnK_old = np.log(np.maximum(K, 1e-30))
        err = np.max(np.abs(lnK_new-lnK_old))
        # Damping improves robustness near critical conditions.
        K = np.exp(0.5*lnK_old + 0.5*lnK_new)
        f0, f1 = rachford_rice(0.0, z, K), rachford_rice(1.0, z, K)
        if f0*f1 > 0:
            raise RuntimeError("Flash iteration moved to a single-phase state; retry with another initialisation")
        beta = brentq(rachford_rice, 0.0, 1.0, args=(z, K), xtol=1e-12)
        if err < tol:
            x = z/(1.0 + beta*(K-1.0)); y = K*x
            x /= x.sum(); y /= y.sum()
            rhoL, ZL = liquid_density_eos(P, T, x, components, kij)
            rhoV, ZV = gas_density(P, T, y, components, kij)
            return {"phase": "two_phase", "vapor_fraction": beta, "x": x, "y": y,
                    "K": K, "ZL": ZL, "ZV": ZV, "rhoL_lbm_ft3": rhoL,
                    "rhoV_lbm_ft3": rhoV, "iterations": iteration+1}
    raise RuntimeError(f"Flash did not converge in {maxiter} iterations")



def _bubble_dew_residual(P: float, T: float, z: np.ndarray,
                         components: Sequence[Component], kind: str, kij=None,
                         inner_tol: float = 1e-9, inner_maxiter: int = 300) -> float:
    """EOS saturation residual: sum(z_i K_i)-1 (bubble) or
    sum(z_i/K_i)-1 (dew), with incipient phase composition iterated.
    """
    n = len(components)
    Tc = np.array([c.Tc for c in components]); Pc = np.array([c.Pc for c in components])
    omega = np.array([c.omega for c in components])
    K = np.exp(np.clip(np.log(Pc/P) + 5.373*(1.0+omega)*(1.0-Tc/T), -30, 30))
    if kind == "bubble":
        x = z.copy()
        for _ in range(inner_maxiter):
            y = K*x
            y /= y.sum()
            Knew = equilibrium_K(P, T, x, y, components, kij)
            err = np.max(np.abs(np.log(np.maximum(Knew, 1e-30)/np.maximum(K, 1e-30))))
            K = np.sqrt(K*Knew)
            if err < inner_tol:
                break
        return float(np.dot(z, K)-1.0)
    elif kind == "dew":
        y = z.copy()
        for _ in range(inner_maxiter):
            x = y/K
            x /= x.sum()
            Knew = equilibrium_K(P, T, x, y, components, kij)
            err = np.max(np.abs(np.log(np.maximum(Knew, 1e-30)/np.maximum(K, 1e-30))))
            K = np.sqrt(K*Knew)
            if err < inner_tol:
                break
        return float(np.dot(z, 1.0/K)-1.0)
    raise ValueError("kind must be 'bubble' or 'dew'")


def saturation_pressure(T: float, z: Sequence[float],
                        components: Sequence[Component], kind: str,
                        kij=None, pressure_bounds=(5.0, 5000.0),
                        scan_points: int = 100) -> float:
    """Find bubble- or dew-point pressure with PR fugacity equality.

    T is deg R; pressure bounds are psia. A logarithmic scan locates a
    sign-changing bracket, then SciPy brentq solves the saturation condition.
    Multiple roots can occur for complex fluids; this returns the first
    bracket encountered in ascending pressure and should be checked against
    experimental PVT data.
    """
    z = _as_composition(z, len(components), "overall composition")
    kind = kind.lower()
    if kind not in ("bubble", "dew"):
        raise ValueError("kind must be 'bubble' or 'dew'")
    pmin, pmax = pressure_bounds
    if pmin <= 0 or pmax <= pmin:
        raise ValueError("pressure_bounds must satisfy 0 < lower < upper")
    pressures = np.geomspace(pmin, pmax, scan_points)
    vals = []
    for p in pressures:
        try:
            vals.append(_bubble_dew_residual(p, T, z, components, kind, kij))
        except (RuntimeError, ValueError, FloatingPointError):
            vals.append(np.nan)
    for i in range(len(pressures)-1):
        a, b = vals[i], vals[i+1]
        if np.isfinite(a) and np.isfinite(b) and a*b <= 0:
            return float(brentq(lambda pp: _bubble_dew_residual(
                pp, T, z, components, kind, kij), pressures[i], pressures[i+1],
                xtol=1e-7, rtol=1e-9, maxiter=200))
    raise RuntimeError(
        f"Could not bracket {kind}-point pressure in {pressure_bounds} psia. "
        "Expand bounds, check component data/BIPs, or use a suitable initialisation.")

def example_c1_components():
    """Appendix C Table C-1 data (Example C-1)."""
    return [
        Component("C3", Tc=665.7, Pc=616.3, omega=0.1454, Mw=44.09, Tb=416.0),
        Component("n-C4", Tc=765.3, Pc=550.7, omega=0.1928, Mw=58.12, Tb=490.8),
        Component("n-C5", Tc=845.4, Pc=488.6, omega=0.2510, Mw=72.15, Tb=556.6),
    ]


if __name__ == "__main__":
    # Basic EOS reproduction check using Example C-1's composition.
    comps = example_c1_components()
    z = [0.6, 0.3, 0.1]
    P, T = 117.7, 100.0 + 459.67  # psia, deg R
    print("Appendix C Example C-1 composition at 117.7 psia and 100 F")
    print("NOTE: Appendix C Example C-1 obtains 117.7 psia with the Hoffman K-value method.")
    for phase in ("liquid", "vapor"):
        phi, Z = pr_fugacity_coefficients(P, T, z, comps, phase)
        print(f"{phase:>6}: Z={Z:.8f}; phi={np.array2string(phi, precision=6)}")
    print("PR liquid density:", liquid_density_eos(P, T, z, comps)[0], "lbm/ft^3")
    print("PR vapor density: ", gas_density(P, T, z, comps)[0], "lbm/ft^3")
