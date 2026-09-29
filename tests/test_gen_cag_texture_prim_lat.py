from gen_cag_texture_prim_lat import expand


def test_expand_matches_cag_style_keys():
    lat = {"mp-1": [1, 2, 3], "mp-2": [4, 5, 6]}
    xrd = {"mp-1_0": None, "mp-1_1": None, "mp-2_0": None}

    out = expand(xrd, lat, "_", "cag")

    assert out == {"mp-1_0": [1, 2, 3], "mp-1_1": [1, 2, 3], "mp-2_0": [4, 5, 6]}


def test_expand_matches_texture_style_keys():
    lat = {"mp-1": [1, 2, 3]}
    xrd = {"mp-1_texture_1": None, "mp-1_texture_2": None}

    out = expand(xrd, lat, "_texture_", "texture")

    assert out == {"mp-1_texture_1": [1, 2, 3], "mp-1_texture_2": [1, 2, 3]}


def test_expand_skips_keys_with_no_lattice_match(capsys):
    lat = {"mp-1": [1, 2, 3]}
    xrd = {"mp-1_0": None, "mp-unknown_0": None}

    out = expand(xrd, lat, "_", "cag")

    assert out == {"mp-1_0": [1, 2, 3]}
    assert "mp-unknown_0" not in out
    assert "WARNING" in capsys.readouterr().out
