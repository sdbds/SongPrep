import argparse
import torch
import torchaudio
import ffmpeg
import numpy as np
from transformers import AutoModelForCausalLM
from megatron.tokenizer import build_tokenizer
from mucodec.generate_1rvq import Tango
from tqdm import tqdm
from pathlib import Path
import re


class Args:
    def __init__(self):
        pass


class megaInf:
    def __init__(self, model_path, vocal_file, tokenizer="Qwen2Tokenizer", extra_vocab_size=16384):
        args = Args()
        args.vocab_file = vocal_file
        args.load = model_path
        args.extra_vocab_size = extra_vocab_size
        args.patch_tokenizer_type = tokenizer

        self.tokenizer = build_tokenizer(args)
        self.text_offset = len(self.tokenizer.tokenizer.get_vocab())
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True,
            )

    def run(self, audio):
        audio = np.array(audio.to("cpu")).astype(np.int32) + self.text_offset
        sentence_ids = [self.tokenizer.sep_token_id] + audio.tolist() + [self.tokenizer.tokenizer.sep_token_id]
        
        prompt = torch.LongTensor(sentence_ids).to("cuda").unsqueeze(0)
        
        # use tqdm show progress
        with tqdm(total=1, desc="Transcribing", unit="song") as pbar:
            generate_ids = self.model.generate(prompt, do_sample=True, 
                                               top_p=0.1,
                                               temperature=0.1, 
                                               num_return_sequences=1,
                                               eos_token_id=self.tokenizer.eos_token_id, 
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               max_length=8192,
                                               ).squeeze(0).cpu().numpy()
            pbar.update(1)

        # import pdb; pdb.set_trace()
        indices = (generate_ids == self.tokenizer.sep_token_id).nonzero()[0]
        assert len(indices) >= 2, indices
        start = indices[1] + 1
        if len(indices) == 2:
            end = -1
        else:
            end = indices[2] - 1
        return self.tokenizer.detokenize(generate_ids[start:end])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", dest="input_wav")
    parser.add_argument("-q", dest="qwen_ckpt", default="SongPrep-7B/")
    parser.add_argument("-c", dest="codec_ckpt", default="SongPrep-7B/mucodec.safetensors")
    args = parser.parse_args()

    vocal_file = "conf/vocab_type.yaml"
    qwen_path = args.qwen_ckpt
    codec_path = args.codec_ckpt
    wav_path = args.input_wav

    # codec
    tango = Tango(model_path=codec_path)
    src_wave, fs = torchaudio.load(wav_path)
    if (fs != 48000):
        src_wave = torchaudio.functional.resample(src_wave, fs, 48000)
    code = tango.sound2code(src_wave)
    del tango
    torch.cuda.empty_cache()

    # transcription
    maga = megaInf(qwen_path, vocal_file)
    lyric = maga.run(code[0][0])
    
    # save lyric to same name txt file
    wav_path_obj = Path(wav_path)
    wav_dir = wav_path_obj.parent
    wav_name = wav_path_obj.stem
    txt_path = wav_dir / f"{wav_name}.txt"
    # format lyric, add newline
    formatted_lyric = lyric.replace(" ; ", ";\n").replace("]", "]\n")
    # add newline after period in English
    formatted_lyric = re.sub(r'([a-zA-Z])\.', r'\1.\n', formatted_lyric).strip()
    print(formatted_lyric)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(formatted_lyric)
    print(f"Lyric saved to: {txt_path}")
    