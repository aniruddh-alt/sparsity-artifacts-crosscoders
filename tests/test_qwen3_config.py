from tools.configs import MODEL_CONFIGS


def test_qwen3_models_registered():
    for model in [
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B",
        "/data/aniruddhan/models/qwen3-0.6b-custom-ft",
    ]:
        assert model in MODEL_CONFIGS, f"missing {model}"
        assert MODEL_CONFIGS[model]["text_column"] == "text_qwen3"
