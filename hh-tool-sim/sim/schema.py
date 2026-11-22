"""What a cell, a protocol and a paper extraction *are*.

Pure data (Pydantic v2). Nothing here integrates equations or reads files.
Units are the model units used by the simulator; each field documents its own.

Referenced from: https://github.com/openworm/hodgkin_huxley_tutorial/blob/master/Tutorial/Source/HodgkinHuxley.py 
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --------------------------------------------------------------------------
# Cell
# --------------------------------------------------------------------------

class HHCell(BaseModel):
    """Classic Hodgkin-Huxley membrane patch. Defaults are the squid-axon values."""

    model_config = ConfigDict(extra="forbid")

    # le=1000 on conductances: a sanity check against unit mistakes (e.g. S/m^2 vs mS/cm^2).
    C_m: float = Field(1.0, gt=0, description="Membrane capacitance, uF/cm^2")
    gmax_Na: float = Field(120.0, ge=0, le=1000, description="Max Na conductance, mS/cm^2")
    gmax_K: float = Field(36.0, ge=0, le=1000, description="Max K conductance, mS/cm^2")
    gmax_L: float = Field(0.3, ge=0, le=1000, description="Leak conductance, mS/cm^2")
    E_Na: float = Field(50.0, description="Na reversal potential, mV")
    E_K: float = Field(-77.0, description="K reversal potential, mV")
    E_L: float = Field(-54.387, description="Leak reversal potential, mV")


# Model unit for each HHCell field. Extracted values must arrive in exactly these units.
HH_UNITS = {
    "C_m": "uF/cm^2",
    "gmax_Na": "mS/cm^2",
    "gmax_K": "mS/cm^2",
    "gmax_L": "mS/cm^2",
    "E_Na": "mV",
    "E_K": "mV",
    "E_L": "mV",
}


# --------------------------------------------------------------------------
# Protocols
#
# Each protocol is its own class, picked from JSON by its `kind` field.
# The simulator only ever calls `I_inj(t, V)` and `switch_times()`, so it
# never needs to know which protocol it was given. A new protocol = a new class
# here (added to the `Protocol` union) plus a summariser in analysis.py.
# --------------------------------------------------------------------------

class ProtocolBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_n: float = Field(gt=0, description="End of simulation (total length), ms")
    delta_t: float = Field(gt=0, description="Output sample interval, ms")
    temperature: float = Field(6.3, description="Temperature, degC (HH kinetics are defined at 6.3)")


class CurrentClamp(ProtocolBase):
    """Single rectangular current pulse."""

    kind: Literal["iclamp"]
    delta_t: float = Field(0.01, gt=0, description="Output sample interval, ms")
    I_inj_amplitude: float = Field(description="Pulse amplitude, uA/cm^2")
    I_inj_delay: float = Field(ge=0, description="Pulse start, ms")
    I_inj_duration: float = Field(ge=0, description="Pulse width, ms")

    @model_validator(mode="after")
    def _delay_before_t_n(self):
        if self.I_inj_delay >= self.t_n:
            raise ValueError(f"I_inj_delay ({self.I_inj_delay}) must be < t_n ({self.t_n})")
        return self

    def I_inj(self, t, V):
        """Injected current density, uA/cm^2. Same expression as the tutorial's I_inj.

        Works for scalar or array `t`; `V` is unused here, only required for voltage clamp. Simulate() takes both by default.
        """
        return self.I_inj_amplitude * (t > self.I_inj_delay) - self.I_inj_amplitude * (t > self.I_inj_delay + self.I_inj_duration) #boolean, returns rectangular pulse 

    def switch_times(self) -> list[float]:
        return [self.I_inj_delay, self.I_inj_delay + self.I_inj_duration] # lists time of pulse on/pulse off 


class VoltageClamp(ProtocolBase):
    """Conditioning -> test -> return voltage steps.

    This is a series-resistance approximation, not an ideal clamp: the clamp
    injects current proportional to (target - V), so V only tracks the target
    approximately and the equations are stiff (hence the tiny default delta_t).
    An "ideal clamp" (fix V to the target, integrate only the gates) is a
    possible later alternative and would be a separate protocol class.
    """

    kind: Literal["vclamp"]
    t_n: float = Field(50.0, gt=0, description="End of simulation, ms")
    delta_t: float = Field(0.0005, gt=0, description="Output sample interval, ms")
    vc_delay: float = Field(10.0, ge=0, description="Start of test step, ms")
    vc_duration: float = Field(30.0, ge=0, description="Length of test step, ms")
    vc_condVoltage: float = Field(-65.0, description="Target before the step, mV")
    vc_testVoltage: float = Field(10.0, description="Target during the step, mV")
    vc_returnVoltage: float = Field(-65.0, description="Target after the step, mV")
    simpleSeriesResistance: float = Field(1e7, gt=0, description="Clamp series resistance (tutorial units)")
    area_um2: float = Field(1000.0, gt=0, description="Assumed membrane area, um^2 (tutorial value)")

    def I_inj(self, t, V):
        """Clamp current density, uA/cm^2. Same logic and arithmetic as the tutorial's I_inj_vclamp.

        Scalar `t` and `V` only (the boundaries use if/elif, as in the tutorial).
        """
        if t > self.vc_delay + self.vc_duration:
            target = self.vc_returnVoltage
        elif t >= self.vc_delay:
            target = self.vc_testVoltage
        else:
            target = self.vc_condVoltage

        current_A = (target - V) / self.simpleSeriesResistance
        current_uA = current_A * 10**6             # A -> uA
        surface_area = self.area_um2 * 10**-8      # um^2 -> cm^2
        return current_uA / surface_area

    def switch_times(self) -> list[float]:
        # Empty on purpose: matches the tutorial, which effectively runs vclamp with no
        # tcrit. Passing [vc_delay, vc_delay + vc_duration] as tcrit makes odeint fail on the
        # clamp's stiff jumps (see the parked odeint_piecewise in hh.py).
        return []


Protocol = Annotated[Union[CurrentClamp, VoltageClamp], Field(discriminator="kind")]


# --------------------------------------------------------------------------
# Paper extraction: the contract the RAG side will later produce.
# --------------------------------------------------------------------------

class ExtractedValue(BaseModel): # parameter value retrieved from a paper, plus evidence for it
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    quote: str = Field(description="Verbatim text from the paper supporting the value")
    location: str = Field(description="Where in the paper, e.g. 'p. 518, Table 3'")
    status: Literal["reported", "derived", "defaulted"]


class ReportedOutcome(BaseModel): # result reported by the paper, to check the simulation against 
    model_config = ConfigDict(extra="forbid")

    measure: Literal[
        "num_spikes", "firing_rate_hz", "rest_mv", "peak_mv",
        "peak_inward_uA_cm2", "peak_outward_uA_cm2",
    ]
    value: ExtractedValue
    condition: str = Field(description="Experimental condition the outcome was reported under")


class PaperExtraction(BaseModel): #written JSON file (LLM reads chunks + fills in JSON schema)
    model_config = ConfigDict(extra="forbid")

    paper: str
    model_type: Literal["classic_hh"]
    notes: str = ""
    cell: dict[str, ExtractedValue]
    protocol: Protocol
    reported_outcomes: list[ReportedOutcome] = []


# --------------------------------------------------------------------------
# Normalise: extraction -> simulator input
# --------------------------------------------------------------------------

def to_hh_cell(extraction: PaperExtraction) -> tuple[HHCell, list[str]]:
    """Build an HHCell from extracted values.

    Returns the cell and the list of fields that were defaulted: fields missing
    from the extraction, plus any the extraction itself marked "defaulted".
    Raises ValueError for unknown fields or units that differ from HH_UNITS.

    KNOWN ISSUE (resolve before stage B):
        Missing parameters fall back to the HHCell defaults, which are the
        squid-axon values from Hodgkin & Huxley 1952. That is only appropriate
        for classic-HH papers. For C. elegans / multi-channel models, a missing
        parameter should probably be an error, or require explicit
        confirmation, rather than silently using squid values. The returned
        list makes the fallback visible but does not prevent it.
    """
    unknown = set(extraction.cell) - set(HHCell.model_fields) # Lists all parameters that fall back to default 
    if unknown:
        raise ValueError(f"Unknown HHCell field(s) in extraction: {sorted(unknown)}")

    values = {}
    for name, ev in extraction.cell.items():
        if ev.unit != HH_UNITS[name]:
            # TODO(stage B): unit conversion goes here
            raise ValueError(
                f"{name}: unit '{ev.unit}' differs from model unit '{HH_UNITS[name]}' "
                "(unit conversion not implemented)"
            )
        values[name] = ev.value

    # TODO(stage B): defaults policy. Missing fields silently take squid-axon defaults here.
    missing = [f for f in HHCell.model_fields if f not in values]
    marked = [n for n, ev in extraction.cell.items() if ev.status == "defaulted"]
    return HHCell(**values), missing + marked
