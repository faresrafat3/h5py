"""Regression tests for the exceptions h5py's high-level API raises.

These lock in the behaviour that the docstrings now describe. If a future
change makes a call raise a different type, the docstring is wrong and this
test fails.

See gh-1447: exceptions were not documented anywhere.
"""

import io
import os
import tempfile

import numpy as np
import pytest

import h5py


@pytest.fixture
def h5file(tmp_path):
    """A file with a dataset, a group, an empty dataset, and an attribute."""
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("d", data=np.arange(10))
        f.create_dataset("empty", data=np.array([]))
        f.attrs["a"] = 1
        f.create_group("g")
    with h5py.File(path, "r") as f:
        yield f


# --------------------------------------------------------------------- File


def test_file_missing_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        h5py.File(tmp_path / "nope.h5", "r")


def test_file_bad_mode_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="Invalid mode"):
        h5py.File(tmp_path / "t.h5", "z")


def test_file_unknown_driver_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="Unknown driver"):
        h5py.File(tmp_path / "t.h5", "w", driver="definitely-not-a-driver")


def test_file_mode_x_on_existing_raises_fileexists(tmp_path):
    p = tmp_path / "t.h5"
    p.touch()
    with pytest.raises(FileExistsError):
        h5py.File(p, "x")


def test_file_mode_r_plus_missing_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        h5py.File(tmp_path / "nope.h5", "r+")


def test_swmr_mode_cannot_be_disabled(tmp_path):
    # Setting the property to False raises; SWMR cannot be turned off.
    with h5py.File(tmp_path / "t.h5", "w") as f:
        with pytest.raises(ValueError):
            f.swmr_mode = False


# -------------------------------------------------------------------- Group


def test_group_get_missing_returns_none(h5file):
    # get() is the non-raising accessor; [] is the raising one.
    assert h5file["g"].get("missing") is None


def test_group_create_group_bad_name_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="name should be string or bytes"):
        h5file.create_group(123)


def test_group_require_group_bad_name_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="name should be string or bytes"):
        h5file.require_group(123)


def test_group_copy_bad_name_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="name should be string or bytes"):
        h5file.copy(h5file["d"], 123)


def test_require_dataset_incompatible_object_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="Incompatible object"):
        h5file.require_dataset("g", (3,), "i8")


def test_require_dataset_shape_mismatch_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="Shapes do not match"):
        h5file.require_dataset("d", (99,), "i8")


# ------------------------------------------------------------------ Dataset


def test_read_direct_on_empty_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="broadcast"):
        h5file["empty"].read_direct(np.zeros(1))


def test_astype_bad_type_raises_typeerror(h5file):
    with pytest.raises(TypeError):
        h5file["d"].astype(12345)


def test_asstr_on_int_dataset_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="string datatype"):
        h5file["d"].asstr()


def test_resize_non_chunked_raises_typeerror(tmp_path):
    with h5py.File(tmp_path / "rw.h5", "w") as f:
        plain = f.create_dataset("plain", data=np.arange(5))
        with pytest.raises(TypeError, match="Only chunked datasets"):
            plain.resize((10,))


def test_resize_negative_shape_raises_overflowerror(tmp_path):
    # Not raised by h5py itself: the value reaches the C layer, which
    # rejects it. Documented because a caller cannot predict it.
    with h5py.File(tmp_path / "rw.h5", "w") as f:
        ds = f.create_dataset("c", data=np.arange(5), chunks=(2,))
        with pytest.raises(OverflowError):
            ds.resize((-1,))


# --------------------------------------------------------- AttributeManager


def test_attrs_create_conflicting_shape_raises_valueerror(h5file):
    with pytest.raises(ValueError, match="conflicts with shape"):
        h5file.attrs.create("a", [1, 2, 3], shape=(99,))


def test_attrs_create_bad_name_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="name should be string or bytes"):
        h5file.attrs.create(123, 1)


def test_attrs_modify_bad_name_raises_typeerror(h5file):
    with pytest.raises(TypeError, match="name should be string or bytes"):
        h5file.attrs.modify(123, 1)


# ----------------------------------------------------------------- Datatype


def test_datatype_from_non_type_id_raises_valueerror(tmp_path):
    with h5py.File(tmp_path / "t.h5", "w") as f:
        with pytest.raises(ValueError, match="is not a TypeID"):
            h5py.Datatype(f.id)


# ------------------------------------------------------------------- errors


def test_errors_module_has_silence_helpers():
    from h5py import _errors

    assert callable(_errors.silence_errors)
    assert callable(_errors.unsilence_errors)


# ------------------------------------------------------------ VirtualSource


def test_virtual_source_dataset_with_other_args_raises_typeerror(tmp_path):
    with h5py.File(tmp_path / "t.h5", "w") as f:
        ds = f.create_dataset("d", data=np.arange(3))
        with pytest.raises(TypeError, match="no other arguments"):
            h5py.VirtualSource(ds, name="x")


def test_virtual_source_path_without_name_raises_typeerror(tmp_path):
    with pytest.raises(TypeError, match="name parameter is required"):
        h5py.VirtualSource("some.h5", shape=(3,))


def test_virtual_source_path_without_shape_raises_typeerror(tmp_path):
    with pytest.raises(TypeError, match="shape parameter is required"):
        h5py.VirtualSource("some.h5", name="d")


# ------------------------------------------------------------ Group.get


def test_group_get_bad_elink_mode_raises_runtimeerror(h5file):
    with pytest.raises(RuntimeError, match="Unsupported link access mode"):
        h5file["g"].get("d", elink_mode="zzz")


def test_group_get_bad_elink_locking_raises_valueerror(h5file):
    with pytest.raises(ValueError, match="Unsupported locking value"):
        h5file["g"].get("d", elink_locking="zzz")


# ------------------------------------------------------------ in_memory


def test_file_in_memory_rejects_driver(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        h5py.File.in_memory(driver="core")


# ------------------------------------------------------------ len()


def test_dataset_len_scalar_raises_typeerror(tmp_path):
    with h5py.File(tmp_path / "t.h5", "w") as f:
        scalar = f.create_dataset("s", data=1)
        with pytest.raises(TypeError):
            scalar.len()
