"""Our simulator vs the OpenWorm tutorial (reference/HodgkinHuxley.py).

The two should match exactly at 6.3 degC, where our temperature factor phi == 1.0
(the tutorial has no temperature).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import odeint

sys.path.insert(0, str(Path(__file__).parents[1] / "reference"))
from HodgkinHuxley import HodgkinHuxley  # noqa: E402

from sim import hh  # noqa: E402
from sim.schema import CurrentClamp, HHCell, VoltageClamp  # noqa: E402

# The tutorial's initial V, m, h, n values (its simulate() default).
INIT = [-64.99584, 0.05296, 0.59590, 0.31773]

CASES = {
    "iclamp": (
        dict(I_inj_amplitude=10, I_inj_delay=50, I_inj_duration=100, t_n=200, delta_t=0.01),
        CurrentClamp(kind="iclamp", I_inj_amplitude=10, I_inj_delay=50, I_inj_duration=100, t_n=200),
    ),
    "vclamp": (
        dict(runMode="vclamp", t_n=50, delta_t=0.0005),
        VoltageClamp(kind="vclamp"),
    ),
}


@pytest.mark.parametrize("kind", CASES)
def test_simulation_match(kind):
    """Compare simulation results, V, m, h, n should match.."""
    ref_kwargs, protocol = CASES[kind]
    ref = HodgkinHuxley(**ref_kwargs)

    # Reference tutorial
    if kind == "iclamp":
        expected = ref.simulate(init_values=INIT)
    else:
        # vclamp: simulate() passes the iclamp pulse times (default 0) as tcrit, which scipy
        # rejects, so it returns a broken trace. Call odeint on the tutorial's dALLdt without tcrit.
        expected = odeint(HodgkinHuxley.dALLdt, INIT, ref.t, args=(ref,)) #FIX LATER 
    # My simulation
    trace = hh.simulate(HHCell(), protocol, init_values=INIT)

    assert np.array_equal(trace.t, ref.t)
    for i, name in enumerate(["V", "m", "h", "n"]):
        assert np.array_equal(getattr(trace, name), expected[:, i]), name


def test_injected_current_match():
    """Injected current must match the tutorial exactly, including just before/at/after each switch time."""
    ic_ref = HodgkinHuxley(**CASES["iclamp"][0])
    ic = CASES["iclamp"][1]
    vc_ref = HodgkinHuxley(**CASES["vclamp"][0])
    vc = CASES["vclamp"][1]
    for t in [0, 49.99, 50, 50.01, 100, 149.99, 150, 150.01, 199]:
        assert ic.I_inj(t, -60.0) == ic_ref.I_inj(t) #dummy voltage, ignored in current clamp
    for t in [0, 9.999, 10, 10.001, 25, 39.999, 40, 40.001, 49]:
        for V in [-70.0, -65.0, 0.0, 10.0]:
            assert vc.I_inj(t, V) == vc_ref.I_inj_vclamp(t, V)
