"""Mixed-length decoding must match independent single-sample decoding."""
import unittest
from types import SimpleNamespace

import torch

from models.multimodal_model import MiniVLM


class NextTokenLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 1)
        with torch.no_grad():
            self.embedding.weight[:, 0] = torch.arange(100)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache=False):
        # Position t always predicts the integer token following the input at t.
        next_ids = inputs_embeds[..., 0].round().long() + 1
        logits = torch.full((*next_ids.shape, 100), -100.0)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 100.0)
        return SimpleNamespace(logits=logits)


class GenerationBatchingTests(unittest.TestCase):
    def test_mixed_answer_starts_match_single_sample_generation(self):
        model = MiniVLM.__new__(MiniVLM)
        torch.nn.Module.__init__(model)
        model.llm = NextTokenLM()
        model.projector = torch.nn.Linear(1, 1)
        model.device_ = torch.device("cpu")
        model.image_token_id = 99
        model.tokenizer = SimpleNamespace(eos_token_id=3)
        # Row 0's original answer token 9 occupies row 1's extra prefix slot.
        # It must never enter row 0's generation window.
        ids = torch.tensor([[99, 0, 9, 3, 3], [99, 0, 0, 9, 3]])
        mask = torch.ones_like(ids)
        visual = torch.zeros((2, 1, 1))
        kwargs = dict(input_ids=ids, attention_mask=mask, visual_features=visual,
                      answer_start=torch.tensor([2, 3]), max_new_tokens=4)
        solo, solo_lengths = model.generate(**kwargs, micro_batch=1)
        batched, batch_lengths = model.generate(**kwargs, micro_batch=2)
        self.assertEqual(solo_lengths, [3, 3])
        self.assertEqual(batch_lengths, [3, 3])
        self.assertEqual(batched[:, :3].tolist(), [[1, 2, 3], [1, 2, 3]])
        self.assertTrue(torch.equal(solo, batched))


if __name__ == "__main__":
    unittest.main()
