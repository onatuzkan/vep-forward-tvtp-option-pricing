"""Frozen-parameter loader edge cases, especially placeholder detection."""
from __future__ import annotations

import textwrap

import pytest

from pde_option_model.params_frozen import (
    _provenance_indicates_placeholder,
    load_frozen_parameters,
)


BASE_BLOB = textwrap.dedent("""\
    scale_P: 282.48
    phi: 0.999614845634
    kappa_per_hour: 0.000385228557
    sigma_y_normal: 0.0035348070
    sigma_y_stress: 0.0924066544
    tvtp:
      alpha01: -1.0156656066
      gamma01: -0.5837780825
      alpha10: -1.8952294098
      gamma10:  0.0776983936
      covariate: RD_WS_standardized_lag1h
      placeholder: false
    pi_filtered: [0.931977, 0.068023]
    spot_price_TRY_MWh: 2917.78
    valuation_utc: "2025-12-31T20:00:00+00:00"
    covariate_lag_hours: 1.0
""")


@pytest.mark.parametrize("text,expected", [
    ("estimated: M9 parameter_estimates.csv sigma1", False),
    ("derived: root-finding on the historical z", False),
    ("assumed: preprocessing metadata; trusted on faith", False),
    ("placeholder: TVTP coefficient not present in the bundle", True),
    ("The stand-in values encode expected regime durations", True),
    ("STAND-IN VALUES encode expected durations", True),
    ("stand in for the missing intercepts", True),
    ("standin (no hyphen) is also caught", True),
    ("parameter_estimates.csv were NOT INCLUDED", True),
    ("fallback occupancy from the M2 metadata", True),
    ("not present in the bundle", True),
    ("", False),
])
def test_provenance_indicates_placeholder(text, expected):
    assert _provenance_indicates_placeholder(text) is expected


def test_placeholder_detected_from_provenance_text_even_when_flag_is_false(tmp_path):
    """Yaml declares ``tvtp.placeholder: false`` but the provenance string of
    tvtp.alpha01 admits it is a stand-in.  Loader must still flag it."""
    yml = BASE_BLOB + textwrap.dedent("""\
        provenance:
          scale_P: "estimated: preprocessing metadata"
          tvtp.alpha01: "The stand-in values encode expected durations of ~300h"
          tvtp.gamma01: "estimated: M9 transition_coefficients.csv"
          tvtp.alpha10: "estimated: M9 transition_coefficients.csv"
          tvtp.gamma10: "estimated: M9 transition_coefficients.csv"
    """)
    p = tmp_path / "frozen.yaml"
    p.write_text(yml, encoding="utf-8")

    params = load_frozen_parameters(p)
    assert params.has_placeholders is True
    assert "tvtp.alpha01" in params.placeholders
    # The other fields must NOT be spuriously flagged
    assert "tvtp.gamma01" not in params.placeholders
    assert "tvtp.gamma10" not in params.placeholders
    assert "scale_P" not in params.placeholders


def test_placeholder_not_flagged_when_provenance_is_clean(tmp_path):
    """The current production yaml (derived + swapped M9) should NOT trigger
    the keyword-based detector for the TVTP fields."""
    yml = BASE_BLOB + textwrap.dedent("""\
        provenance:
          scale_P: "estimated: preprocessing metadata"
          tvtp.alpha01: "derived: occupancy/duration-constrained root-finding"
          tvtp.gamma01: "estimated: M9 transition_coefficients.csv, regime-label swapped"
          tvtp.alpha10: "derived: occupancy/duration-constrained root-finding"
          tvtp.gamma10: "estimated: M9 transition_coefficients.csv, regime-label swapped"
    """)
    p = tmp_path / "frozen.yaml"
    p.write_text(yml, encoding="utf-8")

    params = load_frozen_parameters(p)
    assert params.has_placeholders is False
    assert params.placeholders == []


def test_structured_flag_still_triggers_detection(tmp_path):
    """Legacy path: tvtp.placeholder=true with silent provenance must still
    mark all four TVTP fields as placeholders."""
    yml = BASE_BLOB.replace(
        "  placeholder: false",
        "  placeholder: true",
    ) + textwrap.dedent("""\
        provenance:
          scale_P: "estimated"
    """)
    p = tmp_path / "frozen.yaml"
    p.write_text(yml, encoding="utf-8")

    params = load_frozen_parameters(p)
    for k in ("tvtp.alpha01", "tvtp.gamma01", "tvtp.alpha10", "tvtp.gamma10"):
        assert k in params.placeholders, k
