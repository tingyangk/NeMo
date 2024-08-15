#!/bin/bash
set -euo pipefail

batch_size=1
manifest=/media/data/dataset/LibriSpeech/test_other.json

# CTC models
# model_name=nvidia/parakeet-ctc-1.1b
# python examples/asr/speech_to_text_eval.py pretrained_name=$model_name dataset_manifest=$manifest batch_size=$batch_size output_filename=output.jsonl amp=false compute_dtype=bfloat16 use_cer=false num_workers=1 return_hypotheses=false ctc_decoding.strategy=greedy_batch rnnt_decoding.strategy=greedy_batch rnnt_decoding.greedy.loop_labels=false rnnt_decoding.greedy.use_cuda_graph_decoder=true calculate_rtfx=true 


# RNN-T models
model_name=nvidia/parakeet-rnnt-1.1b

# loop_labels
# nsys profile -c cudaProfilerApi \
# python examples/asr/speech_to_text_eval.py pretrained_name=$model_name dataset_manifest=$manifest batch_size=$batch_size output_filename=output.jsonl amp=false compute_dtype=bfloat16 use_cer=false num_workers=1 return_hypotheses=false ctc_decoding.strategy=greedy_batch rnnt_decoding.strategy=greedy_batch calculate_rtfx=true

# loop frames
# nsys profile -c cudaProfilerApi \
# CUDA_VISIBLE_DEVICES="" \
# python examples/asr/speech_to_text_eval.py pretrained_name=$model_name dataset_manifest=$manifest batch_size=$batch_size output_filename=output.jsonl amp=false compute_dtype=bfloat16 use_cer=false num_workers=1 return_hypotheses=false ctc_decoding.strategy=greedy_batch rnnt_decoding.strategy=greedy_batch rnnt_decoding.greedy.loop_labels=false rnnt_decoding.greedy.use_cuda_graph_decoder=false calculate_rtfx=true

# cuda graphs
# sudo nsys profile --gpu-metrics-device=all -c cudaProfilerApi \
# nsys profile -c cudaProfilerApi \
python examples/asr/speech_to_text_eval.py pretrained_name=$model_name dataset_manifest=$manifest batch_size=$batch_size output_filename=output.jsonl amp=false compute_dtype=bfloat16 use_cer=false num_workers=1 return_hypotheses=false ctc_decoding.strategy=greedy_batch rnnt_decoding.strategy=greedy_batch rnnt_decoding.greedy.loop_labels=false rnnt_decoding.greedy.use_cuda_graph_decoder=true calculate_rtfx=true 

# beam search
# python examples/asr/speech_to_text_eval.py pretrained_name=$model_name dataset_manifest=$manifest batch_size=$batch_size output_filename=output.jsonl amp=false compute_dtype=bfloat16 use_cer=false num_workers=1 return_hypotheses=false ctc_decoding.strategy=greedy_batch rnnt_decoding.strategy=tsd calculate_rtfx=true
