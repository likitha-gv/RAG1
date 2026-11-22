"""How a classic Hodgkin-Huxley cell *behaves*: rate functions and simulate().

simulate() takes validated objects, never modifies them, and returns a Trace.
It does not know which protocol it was given: it only calls
protocol.I_inj(t, V) and protocol.switch_times().
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

from sim.schema import HHCell, Protocol


# Rate functions (1/ms), copied exactly from the OpenWorm Hodgkin-Huxley tutorial
# (reference/HodgkinHuxley.py). Modern convention: rest near -65 mV.
# Note alpha_m and alpha_n are 0/0 at exactly V = -40 and V = -55 mV, as in the tutorial.
# Added temperature term when computing dx/dt later on to scale rate functions. Original tutorial doesn't implement this, assumes default.

def alpha_m(V):
    return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

def beta_m(V):
    return 4.0*np.exp(-(V+65.0) / 18.0)

def alpha_h(V):
    return 0.07*np.exp(-(V+65.0) / 20.0)

def beta_h(V):
    return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

def alpha_n(V):
    return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

def beta_n(V):
    return 0.125*np.exp(-(V+65) / 80.0)


# Ionic currents (uA/cm^2). Same arithmetic order as the tutorial, so results match bit for bit.

def I_Na(cell: HHCell, V, m, h):
    return cell.gmax_Na * m**3 * h * (V - cell.E_Na)

def I_K(cell: HHCell, V, n):
    return cell.gmax_K * n**4 * (V - cell.E_K)

def I_L(cell: HHCell, V):
    return cell.gmax_L * (V - cell.E_L)


def steady_state(V: float) -> list[float]:
    """[V, m_inf, h_inf, n_inf] at a fixed voltage."""
    return [
        V,
        alpha_m(V) / (alpha_m(V) + beta_m(V)),
        alpha_h(V) / (alpha_h(V) + beta_h(V)),
        alpha_n(V) / (alpha_n(V) + beta_n(V)),
    ]


@dataclass
class Trace:
    """Simulation output. Times in ms, V in mV, currents in uA/cm^2."""
    t: np.ndarray
    V: np.ndarray
    m: np.ndarray
    h: np.ndarray
    n: np.ndarray
    I_Na: np.ndarray
    I_K: np.ndarray
    I_L: np.ndarray
    I_inj: np.ndarray


# PARKED (not priority): restart odeint at each switch time. Revisit later.
# A single odeint call with tcrit=[delay, delay+duration] fails on the voltage clamp
# (repeated error test failures, then zeros). For now simulate() matches the tutorial:
# one odeint call, and VoltageClamp.switch_times() returns [] (no tcrit), as the
# tutorial effectively runs vclamp. If re-enabled, call it from simulate() in place of odeint.
#
# def odeint_piecewise(f, y0, t, switch_times) -> np.ndarray: #tcrit = switch_times() - spikes would cause issues with odeint
#     """odeint over the grid `t`, restarted at each switch time.
#
#     Why not one odeint call with tcrit=switch_times: LSODA stops at a tcrit
#     point but keeps its step history from before the jump. The voltage clamp's
#     jump (tens of mV through a very small series resistance) then makes the
#     error test fail repeatedly and odeint gives up, returning zeros for the
#     rest of the run. Restarting gives each piece a clean start. tcrit=[b] in
#     each piece stops LSODA evaluating f past the switch.
#
#     Raises RuntimeError if any piece fails (odeint itself only warns).
#     """
#     inner = sorted(s for s in set(switch_times) if t[0] < s < t[-1])
#     bounds = [t[0], *inner, t[-1]]
#     X = np.empty((len(t), len(y0)))
#     X[0] = y0
#     y = y0
#     for a, b in zip(bounds[:-1], bounds[1:]):
#         idx = np.nonzero((t > a) & (t <= b))[0]          # grid points this piece fills
#         ts = np.unique(np.concatenate(([a], t[idx], [b])))
#         Y, info = odeint(f, y, ts, tcrit=[b], full_output=True)
#         if info["message"] != "Integration successful.":
#             raise RuntimeError(f"odeint failed on [{a}, {b}] ms: {info['message']}")
#         X[idx] = Y[np.searchsorted(ts, t[idx])]
#         y = Y[-1]
#     return X
#
#     """NOTE: Reference tutorial passes cuurrent-clamp tcrit values in voltage-clamp mode. So the reference is run improprly / without tcrit."""
#

def simulate(cell: HHCell, protocol: Protocol, V0: float = -65.0,
             init_values: list[float] | None = None) -> Trace:
    """Integrate the HH equations under `protocol`.

    Starts from the steady state at V0 unless `init_values` ([V, m, h, n]) is given.
    """
    t = np.arange(0, protocol.t_n, protocol.delta_t)
    # Q10 = 3 temperature scaling of the gating kinetics (not in the reference tutorial).
    # At 6.3 degC phi == 1.0 exactly, so results equal the tutorial's.
    phi = 3.0 ** ((protocol.temperature - 6.3) / 10.0)
    y0 = steady_state(V0) if init_values is None else init_values

    def derivatives(X, t):
        V, m, h, n = X
        dVdt = (protocol.I_inj(t, V) - I_Na(cell, V, m, h) - I_K(cell, V, n) - I_L(cell, V)) / cell.C_m
        dmdt = phi * (alpha_m(V)*(1.0-m) - beta_m(V)*m)
        dhdt = phi * (alpha_h(V)*(1.0-h) - beta_h(V)*h)
        dndt = phi * (alpha_n(V)*(1.0-n) - beta_n(V)*n)
        return dVdt, dmdt, dhdt, dndt

    # One call, as in the tutorial. tcrit stops the solver at each jump in the current;
    # the voltage clamp returns no switch times (see VoltageClamp.switch_times).
    tcrit = protocol.switch_times() or None
    X, info = odeint(derivatives, y0, t, tcrit=tcrit, full_output=True)
    if info["message"] != "Integration successful.":
        # odeint itself only warns and fills the rest of the trace with zeros.
        raise RuntimeError(f"odeint failed: {info['message']}")
    V, m, h, n = X.T

    return Trace(
        t=t, V=V, m=m, h=h, n=n,
        I_Na=I_Na(cell, V, m, h),
        I_K=I_K(cell, V, n),
        I_L=I_L(cell, V),
        I_inj=np.array([protocol.I_inj(ti, Vi) for ti, Vi in zip(t, V)]),
    )
