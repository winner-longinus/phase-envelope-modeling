# Appendix C Peng–Robinson EOS implementation (SciPy)

This implementation follows the Peng–Robinson compositional EOS in Appendix C, equations C-19 to C-39, of *Applied Multiphase Flow in Pipes and Flow Assurance*.

## Implemented

- PR pure-component parameters: Eqs. C-24 to C-28 (including the heavy-component `m` correlation for acentric factor > 0.49).
- Mixture mixing rules and dimensionless parameters: Eqs. C-20 to C-26 and C-35 to C-38.
- Cubic compressibility-factor roots: Eqs. C-19 and C-34.
- PR component fugacity coefficients and equilibrium K-values: Eqs. C-10 to C-12.
- Isothermal-isobaric two-phase flash using SciPy `brentq` for the Rachford–Rice equation.
- Gas and liquid density calculations: Eqs. C-33 and C-39.
- EOS bubble/dew saturation-pressure solver using fugacity equality and SciPy `brentq`.
- Appendix C Table C-2 non-hydrocarbon/hydrocarbon BIP values and Eq. C-32 methane/C7+ BIP when C7+ and methane critical volumes are provided.

## Not yet implemented

This is not the entire Appendix C compositional-property package. In particular, the following are not included in this first implementation:

- Hoffman K-value correlation (Eq. C-9) and low-pressure Raoult-law K-values.
- C7+ characterisation correlations from Appendix B.
- Alani–Kennedy liquid-density method (Eqs. C-40 onward).
- Lohrenz–Bray–Clark liquid viscosity and gas-viscosity correlations.
- Parachor surface-tension calculation.
- Peneloux volume translation.
- Complete simulator export tables (PVTO/PVTG).

## Dependencies

```bash
pip install numpy scipy
```

## Units

Use field units consistently:

- Pressure: psia
- Temperature: degrees Rankine (°R)
- Critical temperature: °R
- Critical pressure: psia
- Molecular weight: lbm/lbmol
- Critical volume (if used for C1/C7+ BIP): ft³/lbmol
- Density output: lbm/ft³

Convert Fahrenheit to Rankine using `T_R = T_F + 459.67`.

## Quick start

```python
from appendix_c_peng_robinson import (
    Component, pr_fugacity_coefficients, flash,
    gas_density, liquid_density_eos, saturation_pressure
)

components = [
    Component("C1", Tc=343.0, Pc=667.8, omega=0.0115, Mw=16.04),
    Component("n-C4", Tc=765.3, Pc=550.7, omega=0.1928, Mw=58.12),
    Component("n-C5", Tc=845.4, Pc=488.6, omega=0.2510, Mw=72.15),
]
z = [0.70, 0.20, 0.10]
P = 500.0
T = 100.0 + 459.67

result = flash(P, T, z, components)
print(result)

# For a saturation pressure, select kind="bubble" or kind="dew".
pb = saturation_pressure(T, z, components, kind="bubble")
```

## Validation warning

Appendix C Example C-1 reports a bubble-point pressure of 117.7 psia at 100°F using the Hoffman et al. K-value correlation. That value is **not** an automatic validation target for the PR fugacity method: the K-value method is different. A fair validation must compare the PR implementation with an independent PR calculation or laboratory PVT measurements, using the same component properties, binary interaction parameters, and unit conventions.

The implementation should be further tested with the project's actual fluid composition and measured PVT data before use in reservoir decisions or simulator tables.
