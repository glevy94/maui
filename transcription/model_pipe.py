import torch
import pandas as pd
import librosa

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def init_model(chunk_length_s = 30, batch_size = 6, model_id = "openai/whisper-large-v3"):
    device = torch.device("mps")
    torch_dtype = torch.float16

    model_id = model_id

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        
    )

    return pipe


