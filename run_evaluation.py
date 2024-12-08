# ref -> https://huggingface.co/openai/whisper-large-v3
# ref -> reazonspeech doc

import os
import argparse
from dotenv import load_dotenv
import torch
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel, PeftConfig
from src.dataloader import load_datasets_from_config
from src.metrics import MetricsCalculator, TextNormalizer
from loguru import logger
import numpy as np
from dataclasses import dataclass

from reazonspeech.k2.asr import load_model as load_k2_model, transcribe as k2_transcribe, audio_from_numpy as k2_audio_from_array
from reazonspeech.nemo.asr import load_model as load_nemo_model, transcribe as nemo_transcribe, audio_from_numpy as nemo_audio_from_array
from reazonspeech.espnet.asr import load_model as load_espnet_model, transcribe as espnet_transcribe, audio_from_numpy as espnet_audio_from_array
# from reazonspeech.espnet.oneseg import load_model as load_oneseg_model, transcribe as oneseg_transcribe, audio_from_array as oneseg_audio_from_array

MODEL_TYPES = {
    "whisper": "whisper",
    "k2": "reazonspeech-k2",
    "nemo": "reazonspeech-nemo",
    "espnet": "reazonspeech-espnet",
    # "oneseg": "reazonspeech-oneseg"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_TYPES.keys()),
        help="Type of model to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to model directory (required for Whisper models)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (only for Whisper models)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to dataset config YAML file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (only for Whisper models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (only for Whisper models)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save predictions and metrics (optional)",
    )
    return parser.parse_args()

def load_whisper_model(args):
    """Load Whisper model and processor."""
    logger.info(f"Loading Whisper model from {args.model_path}")
    
    config = AutoConfig.from_pretrained(args.model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    if args.lora_path:
        logger.info(f"Loading LoRA weights from {args.lora_path}")
        peft_config = PeftConfig.from_pretrained(args.lora_path)
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    model.config.language = "japanese"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = model.to(args.device)
    
    if args.device == "cuda":
        model.half()
    
    return model, processor, feature_extractor, tokenizer

def load_reazonspeech_model(model_type):
    """Load ReazonSpeech model based on type."""
    logger.info(f"Loading ReazonSpeech model type: {model_type}")
    
    if model_type == "k2":
        return load_k2_model("cuda"), k2_transcribe, k2_audio_from_array
    elif model_type == "nemo":
        return load_nemo_model("cuda"), nemo_transcribe, nemo_audio_from_array
    elif model_type == "espnet":
        return load_espnet_model("cuda"), espnet_transcribe, espnet_audio_from_array
    elif model_type == "oneseg":
        return load_oneseg_model("cuda"), oneseg_transcribe, oneseg_audio_from_array
    else:
        raise ValueError(f"Unknown ReazonSpeech model type: {model_type}")

def save_results(predictions, references, metrics, output_file):
    """Save predictions and metrics to file."""
    import json
    from pathlib import Path
    
    output_path = Path(output_file)
    results = {
        "predictions": predictions,
        "references": references,
        "metrics": metrics
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def run_whisper_inference(args, model, processor, feature_extractor, tokenizer, test_dataset):
    """Run inference using Whisper model."""
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=args.device,
        batch_size=args.batch_size,
        return_timestamps=False,
        chunk_length_s=30,
        stride_length_s=5,
    )
    
    predictions = []
    references = []
    text_normalizer = TextNormalizer()
    
    for i, batch in enumerate(test_dataset.iter(batch_size=args.batch_size)):
        audios = [{"array": sample["array"], "sampling_rate": sample["sampling_rate"]} 
                 for sample in batch["audio"]]
        
        outputs = pipe(audios, batch_size=args.batch_size)
        batch_preds = [output["text"] for output in outputs]
        batch_refs = batch["sentence"]
        
        batch_preds = [text_normalizer.normalize(pred) for pred in batch_preds]
        batch_refs = [text_normalizer.normalize(ref) for ref in batch_refs]
        
        predictions.extend(batch_preds)
        references.extend(batch_refs)
        
        logger.info(f"\nBatch {i+1}")
        for pred, ref in zip(batch_preds, batch_refs):
            logger.info(f"\nPrediction: {pred}")
            logger.info(f"Reference:  {ref}")
            
    return predictions, references

def run_reazonspeech_inference(model, transcribe_fn, test_dataset):
    """Run inference using ReazonSpeech model."""
    predictions = []
    references = []
    text_normalizer = TextNormalizer()
    
    for i, example in enumerate(test_dataset):
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        audio_input = k2_audio_from_array(audio_array, sampling_rate) # lmao assume same interface for all models
        
        try:
            result = transcribe_fn(model, audio_input)
            prediction = result.text
        except Exception as e:
            logger.error(f"Error transcribing audio {i}: {str(e)}")
            prediction = ""
            
        reference = example["sentence"]
        
        prediction = text_normalizer.normalize(prediction)
        reference = text_normalizer.normalize(reference)
        
        predictions.append(prediction)
        references.append(reference)
        
        if (i + 1) % 10 == 0:
            logger.info(f"\nProcessed {i+1} examples")
            logger.info(f"Last prediction: {prediction}")
            logger.info(f"Last reference:  {reference}")
    
    return predictions, references

def main():
    args = parse_args()
    
    logger.info("Loading test dataset")
    test_dataset = load_datasets_from_config(
        args.dataset_config,
        "test",
        16000,
        1.0,
    )
    
    if args.model_type == "whisper":
        if not args.model_path:
            raise ValueError("model_path is required for Whisper models")
        model, processor, feature_extractor, tokenizer = load_whisper_model(args)
        predictions, references = run_whisper_inference(
            args, model, processor, feature_extractor, tokenizer, test_dataset
        )
    else:
        model, transcribe_fn, audio_from_array = load_reazonspeech_model(args.model_type)
        predictions, references = run_reazonspeech_inference(
            model, transcribe_fn, test_dataset
        )
    
    logger.info("\nCalculating metrics...")
    metrics_calculator = MetricsCalculator(
        tokenizer=None if args.model_type != "whisper" else tokenizer,
        do_normalize_eval=True
    )
    metrics = metrics_calculator.compute_metrics_from_predictions(
        predictions=predictions,
        references=references,
    )
    
    logger.info("\nFinal Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    if args.output_file:
        save_results(predictions, references, metrics, args.output_file)

if __name__ == "__main__":
    main() 