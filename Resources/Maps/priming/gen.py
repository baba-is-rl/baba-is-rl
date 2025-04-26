import glob

import pyBaba as baba


def _map_to_baba(word: str) -> str:
    actual_word = word
    if word.startswith("I_"):
        actual_word = f"ICON_{word.split('_')[1]}"
    elif word == "_":
        actual_word = "ICON_EMPTY"

    return str(baba.ObjectType.__members__[actual_word].value)


def to_baba(in_data: str) -> str:
    in_data_lines = in_data.splitlines()
    w_and_h = in_data_lines.pop(0)
    w_str, h_str = w_and_h.split()
    w = int(w_str)
    h = int(h_str)

    assert len(in_data_lines) == h, "height is not correct"

    lines = [line.split() for line in in_data_lines]
    assert all(len(i) == w for i in lines), "width is not correct"

    return "\n".join(
        [w_and_h] + [" ".join(_map_to_baba(word) for word in line) for line in lines]
    )


if __name__ == "__main__":
    for filename in glob.glob("*.lvl"):
        new_filename = filename.removesuffix(".lvl") + ".txt"
        print(f"{filename} -> {new_filename}")
        with open(filename, "r") as f:
            data = f.read()

        with open(new_filename, "w") as f:
            f.write(to_baba(data))
