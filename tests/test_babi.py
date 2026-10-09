from pathlib import Path

from memwmlm.data.babi import answer_vocabulary, load_babi_task
from memwmlm.tasks.babi import BabiVariant, render_babi_prompt


def test_load_babi_resets_story_at_new_episode(tmp_path: Path) -> None:
    dataset = tmp_path / "task.txt"
    dataset.write_text(
        "1 Mary went to the kitchen.\n"
        "2 John went to the garden.\n"
        "3 Where is Mary?\tkitchen\t1\n"
        "1 Sandra went to the office.\n"
        "2 Where is Sandra?\toffice\t1\n",
        encoding="utf-8",
    )

    examples = load_babi_task(dataset)

    assert len(examples) == 2
    assert examples[0].story == ("Mary went to the kitchen.", "John went to the garden.")
    assert examples[1].story == ("Sandra went to the office.",)
    assert answer_vocabulary(examples) == ("kitchen", "office")


def test_prompt_variants_differ_only_by_source_visibility(tmp_path: Path) -> None:
    dataset = tmp_path / "task.txt"
    dataset.write_text("1 Mary went to the kitchen.\n2 Where is Mary?\tkitchen\t1\n", encoding="utf-8")
    example = load_babi_task(dataset)[0]

    full_context = render_babi_prompt(example, BabiVariant.FULL_CONTEXT)
    application_only = render_babi_prompt(example, BabiVariant.APPLICATION_ONLY)

    assert "Mary went to the kitchen." in full_context.prefix
    assert "Mary went to the kitchen." not in application_only.prefix
    assert "Question: Where is Mary?" in full_context.prefix
    assert full_context.target == application_only.target == "kitchen"
