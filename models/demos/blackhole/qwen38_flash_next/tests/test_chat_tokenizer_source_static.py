# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The served tokenizer is the checkpoint's ``tokenizer.json``, on every transformers release.

``tokenizer_config.json`` names ``Qwen2Tokenizer``; a transformers release without the Qwen3.5 tokenizer class rebuilds
that class's backend with its own pre-tokenizer regex (``\\p{L}+`` where the file has ``[\\p{L}\\p{M}]+``), so a
prompt in a script with combining marks (Thai, Devanagari, Bengali, Arabic with tashkeel) splits at every mark into
about twice the tokens the model was trained on and generates in.  The loader reads ``tokenizer.json`` itself and
refuses a backend that is not the file's; these gates pin the ids of a multilingual set to the file's, keep the
English and code ids and every shipped acceptance record's prompt ids as they are, and exercise both refusals."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.chat import (
    TOKENIZER_COMPONENTS,
    Qwen38ChatFormatError,
    Qwen38OfficialChatTemplate,
    load_checkpoint_tokenizer,
)

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
ACCEPTANCE = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "greedy-prompts"

# One sentence per script; the ids are ``tokenizers.Tokenizer.from_file(tokenizer.json)`` on the text, the form the
# model generates in (the multilingual table tool prints this literal).  The first four are the scripts whose words
# carry combining marks: they were 23 / 38 / 29 / 39 tokens under the class-rebuilt backend of transformers 5.12.1.
MULTILINGUAL_PROMPTS = {
    "thai": "นักเรียนไทยเรียนภาษาอังกฤษที่โรงเรียนทุกวันจันทร์",
    "hindi": "विद्यार्थी हिंदी और अंग्रेज़ी दोनों भाषाएँ सीखते हैं।",
    "bengali": "শিক্ষার্থীরা প্রতিদিন বিদ্যালয়ে বাংলা শেখে।",
    "arabic_tashkeel": "اَلطُّلَّابُ يَتَعَلَّمُونَ اللُّغَةَ الْعَرَبِيَّةَ فِي الْمَدْرَسَةِ.",
    "vietnamese": "Học sinh Việt Nam học tiếng Anh ở trường mỗi ngày.",
    "hebrew_niqqud": "הַתַּלְמִידִים לוֹמְדִים עִבְרִית בְּבֵית הַסֵּפֶר.",
    "korean": "학생들은 매일 학교에서 영어를 배웁니다.",
    "japanese": "学生は毎日学校で英語を勉強します。",
    "chinese": "学生每天在学校学习英语。",
    "russian": "Студенты каждый день изучают английский язык в школе.",
    "german": "Die Schüler lernen jeden Tag Englisch in der Schule.",
    "english": "The students learn English at school every day.",
    "code": "def area(radius: float) -> float:\n    return 3.14159 * radius ** 2\n",
}
# fmt: off
MULTILINGUAL_FILE_IDS = {
    "thai": [184374, 149826, 150605, 191954, 148285, 176408, 186344, 187789],
    "hindi": [176650, 222632, 162029, 149979, 42201, 190488, 173229, 42201, 170729, 148953, 151726, 156950, 194007, 42201, 224406, 181651, 149050, 23121, 148931, 153113, 181545, 149983, 153348, 171094, 154845],
    "bengali": [149473, 167176, 237176, 149988, 161316, 187944, 204345, 151175, 148930, 212488, 150688, 166781, 152447, 56915, 221580, 154845],
    "arabic_tashkeel": [5525, 149331, 42190, 159613, 8270, 148515, 68216, 62422, 70783, 160052, 153605, 149331, 157532, 180277, 148784, 159613, 78958, 191556, 205629, 153458, 150765, 151003, 178402, 27004, 175939, 163355, 155572, 159717, 156312, 188130, 13],
    "vietnamese": [232119, 94456, 169932, 28996, 169889, 170233, 159097, 169863, 169915, 170441, 169888, 13],
    "hebrew_niqqud": [44436, 158926, 52684, 158926, 169450, 47633, 169684, 65035, 146, 112, 160251, 146, 112, 148738, 207645, 146, 117, 65035, 169684, 82823, 146, 112, 148738, 150581, 146, 112, 72948, 169684, 48539, 146, 112, 151552, 85484, 169684, 169450, 72948, 146, 113, 151552, 67000, 158926, 149175, 146, 113, 169450, 148703, 146, 114, 48539, 13],
    "korean": [180042, 151478, 182243, 174102, 54581, 178716, 17862, 71165, 36054, 223, 148348, 13],
    "japanese": [96552, 14876, 247237, 96879, 15685, 118716, 28452, 247240, 74688, 1710],
    "chinese": [96552, 98130, 111305, 136300, 1710],
    "russian": [71000, 34723, 150472, 171756, 159095, 167947, 148803, 245853, 179306, 5624, 187604, 13],
    "german": [17634, 160656, 183046, 51378, 12000, 195466, 303, 2607, 179129, 13],
    "english": [760, 4007, 3827, 6163, 506, 2812, 1396, 1834, 13],
    "code": [727, 2982, 58965, 25, 2153, 8, 1411, 2153, 25, 198, 262, 460, 220, 18, 13, 16, 19, 16, 20, 24, 348, 10264, 2972, 220, 17, 198],
}
# fmt: on
# the class-rebuilt backend's regex (transformers' Qwen2 tokenizer): letters without their combining marks
CLASS_REBUILT_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


@pytest.fixture(scope="module")
def template() -> Qwen38OfficialChatTemplate:
    return Qwen38OfficialChatTemplate(CHECKPOINT)


def _encode(tokenizer, text: str) -> list[int]:
    return [int(value) for value in tokenizer(text, add_special_tokens=False).input_ids]


def test_the_served_backend_is_the_checkpoint_tokenizer_json(template: Qwen38OfficialChatTemplate) -> None:
    from tokenizers import Tokenizer

    file = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    assert template.tokenizer.backend_tokenizer.to_str() == file.to_str()
    parts = json.loads(file.to_str())
    assert parts["normalizer"] == {"type": "NFC"}
    split = parts["pre_tokenizer"]["pretokenizers"][0]
    assert split["type"] == "Split" and "[\\p{L}\\p{M}]+" in split["pattern"]["Regex"]
    assert set(TOKENIZER_COMPONENTS) <= set(parts)


def test_multilingual_prompts_encode_as_the_file_does(template: Qwen38OfficialChatTemplate) -> None:
    from tokenizers import Tokenizer

    file = Tokenizer.from_file(str(CHECKPOINT / "tokenizer.json"))
    for name, text in MULTILINGUAL_PROMPTS.items():
        expected = MULTILINGUAL_FILE_IDS[name]
        assert file.encode(text, add_special_tokens=False).ids == expected, name
        assert _encode(template.tokenizer, text) == expected, name
        assert (
            template.tokenizer.decode(expected, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
        )
    # the rendered prompt carries the same ids: the template adds only the role markers around the text
    rendered = template.render([{"role": "user", "content": MULTILINGUAL_PROMPTS["thai"]}], enable_thinking=False)
    ids = rendered.input_ids[0].tolist()
    thai = MULTILINGUAL_FILE_IDS["thai"]
    assert any(ids[i : i + len(thai)] == thai for i in range(len(ids)))


def test_english_and_code_ids_are_the_pinned_ones(template: Qwen38OfficialChatTemplate) -> None:
    for name in ("english", "code", "german", "russian", "chinese", "japanese", "korean"):
        assert _encode(template.tokenizer, MULTILINGUAL_PROMPTS[name]) == MULTILINGUAL_FILE_IDS[name], name
    markers = "<|im_start|>user\n<think>\nx</think>\n<tool_call>y</tool_call><|im_end|>"
    assert _encode(template.tokenizer, markers) == [
        248045,
        846,
        198,
        248068,
        198,
        87,
        248069,
        198,
        248058,
        88,
        248059,
        248046,
    ]


def test_shipped_acceptance_records_keep_their_prompt_and_reply_ids(template: Qwen38OfficialChatTemplate) -> None:
    tokenizer = template.tokenizer
    records = sorted(ACCEPTANCE.glob("prompt-*-greedy.json"))
    assert len(records) == 13
    for path in records:
        document = json.loads(path.read_text(encoding="utf-8"))
        prompt_ids = document["prompt_token_ids"]
        text = tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert _encode(tokenizer, text) == prompt_ids, path.name
        generated = document["generated_token_ids"]
        decoded = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert decoded == document["generated_text"], path.name
        assert _encode(tokenizer, decoded) == generated, path.name


def test_special_token_contract_holds(template: Qwen38OfficialChatTemplate) -> None:
    tokenizer = template.tokenizer
    assert (len(tokenizer), tokenizer.vocab_size) == (248_077, 248_044)
    assert (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id) == (248_046, 248_044, None)
    assert tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>", "<think>", "</think>"]) == [
        248_045,
        248_046,
        248_068,
        248_069,
    ]
    assert tokenizer.chat_template == (CHECKPOINT / "chat_template.jinja").read_text(encoding="utf-8")


def test_a_missing_tokenizer_json_is_refused(tmp_path: Path) -> None:
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(CHECKPOINT / name, tmp_path / name)
    with pytest.raises(Qwen38ChatFormatError, match="unavailable.*tokenizer.json"):  # allow-pytest.raises: host gate
        load_checkpoint_tokenizer(tmp_path)


def test_a_rewritten_backend_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transformers release that rebuilds the backend the way the Qwen2 class does is refused by name."""

    import transformers
    from tokenizers import Regex, pre_tokenizers

    original = transformers.PreTrainedTokenizerFast

    class Rewritten(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(CLASS_REBUILT_REGEX), behavior="isolated", invert=False),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                ]
            )

    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", Rewritten)
    with pytest.raises(Qwen38ChatFormatError, match="pre_tokenizer differ"):  # allow-pytest.raises: host gate
        load_checkpoint_tokenizer(CHECKPOINT)
