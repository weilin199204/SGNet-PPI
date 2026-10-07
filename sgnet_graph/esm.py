import torch


class EsmEmbedder:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device="cpu"):
        from transformers import AutoTokenizer, EsmModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def embed_sequences(self, sequences):
        outputs = []
        for seq in sequences:
            if not seq:
                continue
            inputs = self.tokenizer(seq, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            result = self.model(**inputs)
            hidden = result.last_hidden_state.squeeze(0)[1:-1].detach().cpu()
            outputs.append(hidden)
        if not outputs:
            raise ValueError("No non-empty sequences to embed.")
        return torch.cat(outputs, dim=0)
