"""Tests for the InstrumentOffset value type: its repr, from_focal_plane and its refusals."""

import math
from decimal import Decimal

import numpy as np
import pytest

from fyst_trajectories.exceptions import PointingError
from fyst_trajectories.offsets import (
    InstrumentOffset,
)
from fyst_trajectories.primecam import resolve_offset


class TestInstrumentRotationRepr:
    """``repr`` shows ``instrument_rotation`` only when it is non-zero."""

    def test_repr_without_instrument_rotation(self):
        offset = InstrumentOffset(dx=5.0, dy=3.0, name="Test")
        r = repr(offset)
        assert "instrument_rotation" not in r
        assert "dx=5.0'" in r
        assert "dy=3.0'" in r
        assert "name='Test'" in r

    def test_repr_with_instrument_rotation(self):
        offset = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        r = repr(offset)
        assert "instrument_rotation=15.0" in r


class TestFromFocalPlane:
    """The mm-to-arcmin factory: the plate-scale conversion and the pass-through fields."""

    def test_basic_conversion(self):
        """The factory converts mm to arcmin through the plate scale."""
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=-461.3, plate_scale=13.89)
        # 461.3 mm * 13.89 arcsec/mm / 60 = 106.79 arcmin
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(-106.79, abs=0.01)

    def test_name_passed_through(self):
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, name="TestModule"
        )
        assert offset.name == "TestModule"

    def test_instrument_rotation_passed_through(self):
        offset = InstrumentOffset.from_focal_plane(
            x_mm=0.0, y_mm=-461.3, plate_scale=13.89, instrument_rotation=15.0
        )
        assert offset.instrument_rotation == pytest.approx(15.0)

    def test_zero_position_returns_zero_offset(self):
        offset = InstrumentOffset.from_focal_plane(x_mm=0.0, y_mm=0.0, plate_scale=13.89)
        assert offset.dx == pytest.approx(0.0, abs=1e-10)
        assert offset.dy == pytest.approx(0.0, abs=1e-10)

    def test_consistency_with_manual_calculation(self):
        x_mm, y_mm, plate_scale = 100.0, 200.0, 13.89

        # Manual calculation
        dx_arcmin_manual = x_mm * plate_scale / 60.0
        dy_arcmin_manual = y_mm * plate_scale / 60.0

        # Via factory
        offset = InstrumentOffset.from_focal_plane(x_mm=x_mm, y_mm=y_mm, plate_scale=plate_scale)

        assert offset.dx == pytest.approx(dx_arcmin_manual, abs=1e-10)
        assert offset.dy == pytest.approx(dy_arcmin_manual, abs=1e-10)

    def test_symmetric_positions(self):
        plate_scale = 13.89

        offset_pos = InstrumentOffset.from_focal_plane(
            x_mm=100.0, y_mm=100.0, plate_scale=plate_scale
        )
        offset_neg = InstrumentOffset.from_focal_plane(
            x_mm=-100.0, y_mm=-100.0, plate_scale=plate_scale
        )

        assert offset_neg.dx == pytest.approx(-offset_pos.dx, abs=1e-10)
        assert offset_neg.dy == pytest.approx(-offset_pos.dy, abs=1e-10)


_NON_FINITE = pytest.mark.parametrize(
    "bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "minus-inf"]
)


class TestNonFiniteFieldsAreRefused:
    """A NaN or infinite ``dx``, ``dy`` or ``instrument_rotation`` is refused at construction.

    ``from_focal_plane`` refuses a NaN or infinite argument by its own name.
    Every projection therefore receives finite fields. The request is
    malformed, so the refusal is a plain ``ValueError``, never a
    ``PointingError``.
    """

    @_NON_FINITE
    @pytest.mark.parametrize("field", ["dx", "dy", "instrument_rotation"])
    def test_constructor_refuses(self, field, bad):
        kwargs = {"dx": 5.0, "dy": 3.0, field: bad}
        with pytest.raises(ValueError, match=f"^{field} must be a finite number") as excinfo:
            InstrumentOffset(**kwargs)
        assert not isinstance(excinfo.value, PointingError)

    @_NON_FINITE
    @pytest.mark.parametrize("argument", ["x_mm", "y_mm", "plate_scale", "instrument_rotation"])
    def test_from_focal_plane_refuses(self, argument, bad):
        kwargs = {"x_mm": 100.0, "y_mm": -50.0, "plate_scale": 13.89, argument: bad}
        with pytest.raises(ValueError, match=f"^{argument} must be a finite number"):
            InstrumentOffset.from_focal_plane(**kwargs)

    def test_from_focal_plane_refuses_a_conversion_that_overflows(self):
        # Finite inputs whose arcminute product passes the largest double.
        with pytest.raises(ValueError, match="^dx must be a finite number, got inf"):
            InstrumentOffset.from_focal_plane(x_mm=1e308, y_mm=0.0, plate_scale=13.89)

    @_NON_FINITE
    @pytest.mark.parametrize("field", ["dx", "dy"])
    def test_resolve_offset_refuses(self, field, bad):
        with pytest.raises(ValueError, match=f"^{field} must be a finite number"):
            resolve_offset(**{field: bad})


class TestFieldsAreRealNumbers:
    """``dx``, ``dy`` and ``instrument_rotation`` take a real number and hold it as a float."""

    @pytest.mark.parametrize(
        "bad", [True, np.array([5.0]), "5.0", None], ids=["bool", "array", "text", "none"]
    )
    @pytest.mark.parametrize("field", ["dx", "dy", "instrument_rotation"])
    def test_a_non_number_is_refused_by_name(self, field, bad):
        kwargs = {"dx": 5.0, "dy": 3.0, field: bad}
        with pytest.raises(ValueError, match=f"^{field} must be a real number"):
            InstrumentOffset(**kwargs)

    def test_each_field_is_stored_as_a_float(self):
        """A 0-d array, a ``Decimal`` or an ``int`` is held as the float it equals, so it hashes."""
        offset = InstrumentOffset(dx=np.array(5.0), dy=Decimal("3.0"), instrument_rotation=15)
        assert [type(offset.dx), type(offset.dy), type(offset.instrument_rotation)] == [float] * 3
        plain = InstrumentOffset(dx=5.0, dy=3.0, instrument_rotation=15.0)
        assert offset == plain
        assert hash(offset) == hash(plain)

    @pytest.mark.parametrize(
        "bad", [True, np.array([5.0]), "5.0", None], ids=["bool", "array", "text", "none"]
    )
    @pytest.mark.parametrize("argument", ["x_mm", "y_mm", "plate_scale"])
    def test_from_focal_plane_refuses_a_non_number_by_name(self, argument, bad):
        """Each is checked before the conversion, which would read a bool as 0 or 1."""
        kwargs = {"x_mm": 100.0, "y_mm": -50.0, "plate_scale": 13.89, argument: bad}
        with pytest.raises(ValueError, match=f"^{argument} must be a real number"):
            InstrumentOffset.from_focal_plane(**kwargs)

    def test_from_focal_plane_converts_a_decimal(self):
        """A ``Decimal`` position and plate scale give the offset of the floats."""
        offset = InstrumentOffset.from_focal_plane(
            x_mm=Decimal("100.0"), y_mm=Decimal("-50.0"), plate_scale=Decimal("13.89")
        )
        plain = InstrumentOffset.from_focal_plane(x_mm=100.0, y_mm=-50.0, plate_scale=13.89)
        assert offset == plain
