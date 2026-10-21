from unsloth.registry.registry import ModelInfo, ModelMeta, _register_models

_IS_MINICPMV_4_7_REGISTERED = False


class MiniCPMVModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)


# unsloth has no MiniCPM-V uploads on the Hub yet, so this family is not wired
# into register_models() (unsloth org models only) and has no QuantType entries.
MiniCPMV_4_7_Meta = ModelMeta(
    org = "openbmb",
    base_name = "MiniCPM-V",
    instruct_tags = [None],  # no separate base/instruct variants in 4.7
    model_version = "4.7",
    model_sizes = ["1B", "35B-A3B"],
    model_info_cls = MiniCPMVModelInfo,
    is_multimodal = True,
    quant_types = [],
)


# include_original_model defaults to True: the openbmb originals are the only
# registration this family has until unsloth uploads MiniCPM-V quants.
def register_minicpmv_models(include_original_model: bool = True):
    global _IS_MINICPMV_4_7_REGISTERED
    if _IS_MINICPMV_4_7_REGISTERED:
        return
    _register_models(MiniCPMV_4_7_Meta, include_original_model = include_original_model)
    _IS_MINICPMV_4_7_REGISTERED = True


if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info

    MODEL_REGISTRY.clear()

    register_minicpmv_models(include_original_model = True)

    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
