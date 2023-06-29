from mmpl.registry import MODELS
from mmengine.model import BaseModule
from transformers import GPT2LMHeadModel, GPT2Config, GPT2PreTrainedModel, PreTrainedModel


@MODELS.register_module()
class HFGPTTransformerLM(BaseModule):
    def __init__(
            self,
            model_name='gpt2',
            from_pretrained=True,
            update_kwargs=dict(
                max_position_embeddings=512,
                hidden_size=512,
            )
    ):
        # BaseModule.__init__(self)
        super().__init__()
        self.model_name = model_name
        if from_pretrained:
            self.gpt_model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            config = GPT2Config.from_pretrained(model_name)
            config.update(update_kwargs)
            self.gpt_model = GPT2LMHeadModel(config=config)

    def forward(self, *args, **kwargs):
        out_puts = self.gpt_model(*args, **kwargs)
        return out_puts

    def generate(self, *args, **kwargs):
        return self.gpt_model.generate(*args, **kwargs)
