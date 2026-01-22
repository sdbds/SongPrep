import os
import argparse
import torch
import torchaudio
from tqdm import tqdm
from vllm.v1.engine.processor import Processor
from vllm.engine.llm_engine import LLMEngine

# Bypass checkout
Processor._validate_model_input = lambda *args, **kwargs: None
LLMEngine._validate_token_prompt = lambda *args, **kwargs: None

from vllm import __version__ as vllm_version
from vllm import LLM, SamplingParams
from megatron.tokenizer import build_tokenizer
from mucodec.generate_1rvq import Tango

class Args:
    def __init__(self):
        pass


class vllmInf:
    def __init__(self, model_path, vocal_file, tokenizer="Qwen2Tokenizer", extra_vocab_size=16384):
        args = Args()
        args.vocab_file = vocal_file
        args.load = model_path
        args.extra_vocab_size = extra_vocab_size
        args.patch_tokenizer_type = tokenizer

        self.tokenizer = build_tokenizer(args)
        self.text_offset = len(self.tokenizer.tokenizer.get_vocab())
        self.max_tokens = 8192
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=self.max_tokens,
            dtype="bfloat16",
        )

    def run(self, audios: list[list[int]]):
        batch_token_ids = []
        max_tokens = self.max_tokens
        for audio in audios:
            audio = audio + self.text_offset
            sentence_ids = [self.tokenizer.sep_token_id] + audio.tolist() + [self.tokenizer.tokenizer.sep_token_id]
            max_tokens = min(max_tokens, self.max_tokens - len(sentence_ids))
            batch_token_ids.append(sentence_ids)

        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            top_p=0.1,
            temperature=0.1
        )

        if vllm_version == "0.8.5":
            outputs = self.llm.generate(
                prompt_token_ids=batch_token_ids,
                sampling_params=sampling_params
            )
        else:
            inputs = [
                {"prompt_token_ids": token_ids}
                for token_ids in batch_token_ids
            ]
            outputs = self.llm.generate(
                prompts=inputs,
                sampling_params=sampling_params
            )

        lyrics = []
        for output in outputs:
            generate_ids = output.outputs[0].token_ids
            lyrics.append(self.tokenizer.detokenize(generate_ids))
        
        return lyrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", dest="input_dir")
    parser.add_argument("-q", dest="qwen_ckpt", default="SongPrep-7B/")
    parser.add_argument("-c", dest="codec_ckpt", default="SongPrep-7B/mucodec.safetensors")
    args = parser.parse_args()

    vocal_file = "conf/vocab_type.yaml"
    qwen_path = args.qwen_ckpt
    codec_path = args.codec_ckpt
    input_dir = args.input_dir

    # codec
    input_audio_paths = []
    input_audios = []
    tango = Tango(model_path=codec_path)
    for audio_path in tqdm(os.listdir(input_dir)):
        if not audio_path.endswith(".wav"):
            continue
        src_wave, fs = torchaudio.load(os.path.join(input_dir, audio_path))
        if (fs != 48000):
            src_wave = torchaudio.functional.resample(src_wave, fs, 48000)
        code = tango.sound2code(src_wave)
        input_audios.append(code[0][0].cpu().numpy())
        input_audio_paths.append(audio_path)
    del tango
    torch.cuda.empty_cache()

    # batch transcription
    vllm_inf = vllmInf(qwen_path, vocal_file)
    lyrics = vllm_inf.run(input_audios)
    
    # display
    for audio_path, lyric in zip(input_audio_paths, lyrics):
        print(f"====={audio_path}=====")
        print(lyric)
        print("\n")
