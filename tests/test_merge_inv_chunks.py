import h5py
import numpy as np

from merge_inv_chunks import merge_chunks


def _write_chunk(path, entries):
    with h5py.File(path, "w") as f:
        for mat_id, pred in entries.items():
            grp = f.create_group(mat_id)
            grp.create_dataset("pred", data=pred)


def test_merge_chunks_combines_disjoint_entries(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_chunk(chunks_dir / "inv_0_1.h5", {"mp-1": np.eye(3)})
    _write_chunk(chunks_dir / "inv_1_2.h5", {"mp-2": np.eye(3) * 2})

    out_path = tmp_path / "merged.h5"
    merge_chunks(str(chunks_dir), str(out_path))

    with h5py.File(out_path, "r") as f:
        assert set(f.keys()) == {"mp-1", "mp-2"}
        np.testing.assert_allclose(f["mp-1"]["pred"][()], np.eye(3))
        np.testing.assert_allclose(f["mp-2"]["pred"][()], np.eye(3) * 2)


def test_merge_chunks_skips_already_merged_duplicates(tmp_path, capsys):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    _write_chunk(chunks_dir / "inv_0_1.h5", {"mp-1": np.eye(3)})

    out_path = tmp_path / "merged.h5"
    merge_chunks(str(chunks_dir), str(out_path))
    merge_chunks(str(chunks_dir), str(out_path))  # re-run against same chunk

    with h5py.File(out_path, "r") as f:
        assert list(f.keys()) == ["mp-1"]
    assert "skipped 1 duplicates" in capsys.readouterr().out
