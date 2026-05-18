from state import chat_session_key


def test_chat_session_key_is_stable_across_model_switches() -> None:
    dataset = "HuggingFace: synth-persona"

    assert chat_session_key("google/gemma-2-2b-it", dataset) == chat_session_key(
        "google/gemma-2-9b-it",
        dataset,
    )


def test_chat_session_key_still_separates_datasets() -> None:
    model = "google/gemma-2-2b-it"

    assert chat_session_key(model, "dataset-a") != chat_session_key(model, "dataset-b")
