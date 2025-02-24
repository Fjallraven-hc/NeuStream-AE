from dataclasses import dataclass

@dataclass
class Model:
    class_name: str
    name: str

@dataclass
class ModelConfig:
    do_predict: bool = False

    model_list: list[Model] | None = None


DefaultModelConfig = ModelConfig(
    do_predict=False,
    model_list=[
        Model(class_name="CodeLlama", name="Code Llama 1"), 
        Model(class_name="LLaVA", name="LLaVA"),
        Model(class_name="CodeLlama", name="Code Llama 2")]
)