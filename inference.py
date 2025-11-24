#!/usr/bin/env python3
#
# Inference script for OpenTSLM models
#

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
import numpy as np
from model.llm.OpenTSLMSP import OpenTSLMSP
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from prompt.text_prompt import TextPrompt
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from prompt.full_prompt import FullPrompt


class OpenTSLMInference:
    """
    Wrapper class for running inference on trained OpenTSLM models.

    Usage:
        inference = OpenTSLMInference(
            model_type="OpenTSLMSP",
            checkpoint_path="results/Llama3_2_1B/OpenTSLMSP/stage2_captioning/checkpoints/best_model.pt",
            llm_id="meta-llama/Llama-3.2-1B"
        )

        time_series = np.random.randn(100)
        response = inference.run(
            pre_prompt="Analyze this time series:",
            time_series_list=[time_series],
            time_series_text_list=["Time series:"],
            post_prompt="What patterns do you observe?",
            max_new_tokens=200
        )
    """

    def __init__(
        self,
        model_type: str = "OpenTSLMSP",
        checkpoint_path: str = None,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = None
    ):
        self.model_type = model_type
        self.device = device or self._get_device()
        self.llm_id = llm_id

        self.model = self._load_model()

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device)
        elif self.model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                device=self.device,
                llm_id=self.llm_id,
                cross_attn_every_n_layers=1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        model.eval()
        return model

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if self.model_type == "OpenTSLMSP":
            self.model.encoder.load_state_dict(checkpoint["encoder_state"])
            self.model.projector.load_state_dict(checkpoint["projector_state"])

            if checkpoint.get("lora_enabled", False):
                self.model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)

        elif self.model_type == "OpenTSLMFlamingo":
            model_state = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(model_state, strict=False)

    def run(
        self,
        pre_prompt: str,
        time_series_list: list,
        time_series_text_list: list,
        post_prompt: str = "",
        max_new_tokens: int = 200
    ) -> str:
        """
        Run inference with custom prompt and time series data.

        Args:
            pre_prompt: Text before the time series
            time_series_list: List of time series arrays (1D numpy arrays or lists)
            time_series_text_list: List of text descriptions for each time series
            post_prompt: Text after the time series
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        if len(time_series_list) != len(time_series_text_list):
            raise ValueError("time_series_list and time_series_text_list must have the same length")

        pre = TextPrompt(pre_prompt)
        post = TextPrompt(post_prompt)

        ts_prompts = []
        for ts, text in zip(time_series_list, time_series_text_list):
            if not isinstance(ts, np.ndarray):
                ts = np.array(ts)
            ts_prompts.append(TextTimeSeriesPrompt(text, ts))

        full_prompt = FullPrompt(pre, ts_prompts, post)

        with torch.no_grad():
            response = self.model.eval_prompt(full_prompt, max_new_tokens=max_new_tokens)

        return response


def main():
    """Example usage"""
    MODEL_TYPE = "OpenTSLMSP"
    LLM_ID = "meta-llama/Llama-3.2-1B"
    CHECKPOINT_PATH = f"results/Llama3_2_1B/{MODEL_TYPE}/stage2_captioning/checkpoints/best_model.pt"

    inference = OpenTSLMInference(
        model_type=MODEL_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
        llm_id=LLM_ID,
        device="cuda"
    )

    # Generate example time series
    time_series = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 0.1

    response = inference.run(
        pre_prompt="Analyze this time series data:",
        time_series_list=[time_series],
        time_series_text_list=["Time series:"],
        post_prompt="What patterns or trends do you observe?",
        max_new_tokens=150
    )

    print(response)


if __name__ == "__main__":
    main()
