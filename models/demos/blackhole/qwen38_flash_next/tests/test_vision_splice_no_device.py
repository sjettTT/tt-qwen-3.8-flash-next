# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host side of the vision splice (``vision_splice.py``) and the decode-tail rule of the rotary positions: no
device, no checkpoint."""

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.mrope import (
    IMAGE_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    Qwen38ImageGrid,
    mrope_positions,
)
from models.demos.blackhole.qwen38_flash_next.vision_splice import (
    HIDDEN_SIZE,
    IMAGE_LANE_SENTINEL_TOKEN,
    NEGATIVE_ZERO_BF16_BITS,
    Qwen38VisionPrompt,
    clean_feature_rows,
    feature_rows_image,
    image_lanes,
    sentinel_token_rows,
    split_features,
)

TEXT = [1000, 1001, 1002, 1003]


def _prompt(image: Qwen38ImageGrid, tail: int):
    return TEXT + [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * image.merged_tokens + [VISION_END_TOKEN_ID] + TEXT[:tail]


def test_sentinel_matches_the_embeddings_zero_row_token():
    # ttnn.embedding's zero sentinel (embedding.py ZERO_EMBEDDING_TOKEN) without importing ttnn here.
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ttnn" / "embedding.py").read_text(encoding="utf-8")
    assert f"ZERO_EMBEDDING_TOKEN = {IMAGE_LANE_SENTINEL_TOKEN}\n" in source


def test_image_lanes_and_sentinel_rows():
    ids = TEXT + [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID] + TEXT
    assert image_lanes(ids) == [4, 5]
    rows = torch.tensor(ids, dtype=torch.float32).reshape(1, 1, 1, -1)
    with_sentinel = sentinel_token_rows(rows, [4, 5])
    assert with_sentinel.reshape(-1)[4:6].tolist() == [-1.0, -1.0]
    assert torch.equal(with_sentinel.reshape(-1)[:4], rows.reshape(-1)[:4])
    assert torch.equal(rows.reshape(-1)[4:6], torch.tensor([float(IMAGE_TOKEN_ID)] * 2))  # the input is untouched
    with pytest.raises(ValueError, match="outside"):
        sentinel_token_rows(rows, [len(ids)])


def test_clean_rows_are_negative_zero_everywhere():
    rows = clean_feature_rows(32, hidden=8)
    assert rows.shape == (1, 1, 32, 8) and rows.dtype == torch.bfloat16
    assert (rows.view(torch.int16) == NEGATIVE_ZERO_BF16_BITS).all()
    assert (rows == 0).all()  # -0.0 == 0.0
    # The identity of the add: x + (-0.0) == x bitwise for +0.0, -0.0 and ordinary values.
    values = torch.tensor([0.0, -0.0, 1.5, -2.25, 1e-3], dtype=torch.bfloat16)
    summed = values + rows[0, 0, 0, :5]
    assert torch.equal(summed.view(torch.int16), values.view(torch.int16))
    with pytest.raises(ValueError):
        clean_feature_rows(0)


def test_feature_rows_place_the_features_at_the_image_lanes():
    ids = TEXT[:2] + [IMAGE_TOKEN_ID] * 3 + TEXT[:3]
    features = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8).to(torch.bfloat16)
    rows = feature_rows_image(ids, features, hidden=8)
    assert rows.shape == (1, 1, 8, 8)
    assert torch.equal(rows[0, 0, 2:5], features)
    text = torch.cat([rows[0, 0, :2], rows[0, 0, 5:]])
    assert (text.view(torch.int16) == NEGATIVE_ZERO_BF16_BITS).all()
    assert feature_rows_image(TEXT, None, hidden=8) is None
    assert feature_rows_image(TEXT, torch.zeros((0, 8), dtype=torch.bfloat16), hidden=8) is None
    with pytest.raises(ValueError, match="3 image pads in the chunk vs 2"):
        feature_rows_image(ids, features[:2], hidden=8)
    with pytest.raises(ValueError, match="BF16"):
        feature_rows_image(ids, features.float(), hidden=8)


def test_split_features_walks_the_chunks():
    features = torch.zeros((5, HIDDEN_SIZE), dtype=torch.bfloat16)
    for row in range(5):
        features[row, 0] = row
    first = [IMAGE_TOKEN_ID] * 2 + TEXT
    second = TEXT + [IMAGE_TOKEN_ID] * 3
    part, cursor = split_features(first, features, 0)
    assert cursor == 2 and part[:, 0].tolist() == [0.0, 1.0]
    none, cursor = split_features(TEXT, features, cursor)
    assert none is None and cursor == 2
    part, cursor = split_features(second, features, cursor)
    assert cursor == 5 and part[:, 0].tolist() == [2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="3 image pads but 0 feature rows remain"):
        split_features(second, features, cursor)
    with pytest.raises(ValueError, match="2 image pads but 0 feature rows remain"):
        split_features(first, None, 0)


def test_shift_at_and_the_decode_tail_rule():
    image = Qwen38ImageGrid(1, 8, 8)  # 16 pads, span 4: shift 12
    ids = _prompt(image, tail=4)
    positions = mrope_positions(ids, [image])
    assert positions.shift == 12
    assert positions.shift_at(positions.length) == 12
    assert positions.shift_at(len(TEXT)) == 0  # before the image: plain text
    assert positions.shift_at(len(TEXT) + 1 + 1) == 0  # the first pad sits at its own index
    assert positions.shift_at(len(TEXT) + 1 + 16) == 12  # after the last pad: the whole shift
    with pytest.raises(ValueError):
        positions.shift_at(positions.length + 1)
    # The tail: <|vision_end|> and four text tokens follow the last pad.  Token length - 1 = index 25, its block
    # starts at 24 (= <|vision_end|> + 3): every token from 24 to 25 is plain text at index - 12.
    assert positions.tail_is_plain(positions.length)
    # A prompt whose last image ends inside the last index block: <|vision_end|> alone after the pads.
    short = mrope_positions(_prompt(image, tail=0), [image])
    end = len(TEXT) + 1 + 16  # the <|vision_end|> index (21); its block start 20 is a pad
    assert not short.tail_is_plain(end + 1)
    # The two decode formulas disagree exactly when the shift is not a multiple of four: (P & ~3) - S is the block's
    # first token's row for a plain tail, (P - S) & ~3 is not.
    P = positions.length + 5
    S = positions.shift
    assert (P & ~3) - S == int(positions.axes[0, P & ~3 - 0].item()) if (P & ~3) < positions.length else True
    odd = Qwen38ImageGrid(1, 6, 6)  # 9 pads, span 3: shift 6
    odd_positions = mrope_positions(_prompt(odd, tail=4), [odd])
    assert odd_positions.shift == 6 and odd_positions.tail_is_plain(odd_positions.length)
    P = odd_positions.length + 2
    block_start_token_row = (
        int(odd_positions.axes[0, (P & ~3)].item()) if (P & ~3) < odd_positions.length else (P & ~3) - 6
    )
    assert (P & ~3) - 6 == block_start_token_row
    assert ((P - 6) & ~3) != (P & ~3) - 6 or (P & ~3) % 4 == 0 and 6 % 4 == 0


def test_tail_rule_keeps_the_shift_within_the_block_start():
    """The device rule S <= P & ~3 (contracts._require_rope_shift) follows from tail_is_plain at the prompt end."""

    for grid in (Qwen38ImageGrid(1, 8, 8), Qwen38ImageGrid(1, 6, 6), Qwen38ImageGrid(1, 2, 30)):
        for tail in (4, 5, 9):
            positions = mrope_positions(_prompt(grid, tail), [grid])
            length = positions.length
            assert positions.tail_is_plain(length)
            for consumed in (length - 1, length):  # the chunks' end and the first decode position
                assert positions.shift_at(consumed) <= consumed & ~3


def test_vision_prompt_validation():
    image = Qwen38ImageGrid(1, 4, 4)
    ids = _prompt(image, tail=4)
    positions = mrope_positions(ids, [image])
    features = torch.zeros((image.merged_tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
    prompt = Qwen38VisionPrompt(positions, features, digest="abc")
    prompt.validate_prompt(ids)
    with pytest.raises(ValueError, match="rotary positions"):
        prompt.validate_prompt(ids + TEXT)
    with pytest.raises(ValueError, match="feature rows"):
        Qwen38VisionPrompt(positions, features[:-1]).validate_prompt(ids)
    with pytest.raises(ValueError, match="BF16"):
        Qwen38VisionPrompt(positions, features.float())
    short_ids = _prompt(image, tail=0)
    short = Qwen38VisionPrompt(mrope_positions(short_ids, [image]), features)
    with pytest.raises(ValueError, match="last index block"):
        short.validate_prompt(short_ids)
