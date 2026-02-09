# tests/test_map_loading.py
import pytest
from shover_world.environment import ShoverWorldEnv


def write_temp(text, rows=5, cols=5):
    path = "temp_map.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


def test_map_loading_malformed_rows():
    path = write_temp("""
0 0 0 0 0
0 0 0 0
""")
    with pytest.raises(ValueError):
        ShoverWorldEnv(n_rows=2, n_cols=5, map_path=path)


def test_map_loading_invalid_token():
    path = write_temp("""
0 0 x 0 0
0 0 0 0 0
""")
    with pytest.raises(ValueError):
        ShoverWorldEnv(n_rows=2, n_cols=5, map_path=path)


def test_map_loading_box_on_edge():
    path = write_temp("""
1 0 0 0 0
0 0 0 0 0
""")
    with pytest.raises(ValueError):
        ShoverWorldEnv(n_rows=2, n_cols=5, map_path=path)
