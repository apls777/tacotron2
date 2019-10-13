import logging
import re
import subprocess
import sys
from shutil import which, rmtree
import numpy as np
from hparams import create_hparams
from text.cleaners import english_cleaners
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
import torch
from waveglow.mel2samp import MAX_WAV_VALUE
import os
from waveglow.denoiser import Denoiser


def convert_book(book_path: str, prefixes_to_fix: list, chunk_limit: int, tacotron_checkpoint_path: str,
                 waveglow_checkpoint_path: str, sampling_rate: int, tmp_chunks_dir: str, output_file_path: str):
    # load the book
    with open(book_path, encoding='utf-8') as f:
        text = f.read()

    # create an output directory
    os.makedirs(tmp_chunks_dir, exist_ok=True)

    # fix prefixes
    text = fix_prefixes(text, prefixes_to_fix)

    # replace new lines with some pauses
    text = text.replace('\n', '. ')

    # clean and normalize the text
    text = english_cleaners(text)

    # split text into chunks
    text_chunks = split_text(text.strip(), chunk_limit)
    logging.info('The book was split to %d chunks' % len(text_chunks))

    # load models
    tacotron = load_tacotron(tacotron_checkpoint_path, sampling_rate)
    waveglow = load_waveglow(waveglow_checkpoint_path)

    denoiser = Denoiser(waveglow)

    audio_paths = []
    for i, text_chunk in enumerate(text_chunks):
        logging.info('Generating audio for the chunk #%d...' % (i + 1))
        logging.info(text_chunk)

        # make spectrogram predictions
        sequence = np.array(text_to_sequence(text_chunk, []))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        _, mel_outputs_postnet, _, _ = tacotron.inference(sequence)

        # synthesize audio
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

        audio = denoiser(audio, strength=0.01)[0, 0]

        audio *= MAX_WAV_VALUE
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')

        audio_path = os.path.join(tmp_chunks_dir, 'chunk_%04d.wav' % i)
        audio_paths.append(audio_path)
        write(audio_path, sampling_rate, audio)

    # merge chunks to one mp3 file
    logging.info('Merging chunks...')
    merge_chunks(audio_paths, output_file_path)

    # remove temporary directory with chunks
    rmtree(tmp_chunks_dir)


def fix_prefixes(text: str, prefixes: list):
    for prefix in prefixes:
        pattern = re.compile(r'\b' + prefix + '([a-z]+)', re.IGNORECASE)
        replacement = prefix + r'-\1'
        text = re.sub(pattern, replacement, text)

    return text


def split_text(text: str, max_num_characters: int):
    chunks = []
    offset = 0
    while offset + max_num_characters < len(text):
        chunk_end = offset + max_num_characters
        for sep in '.!?;,: ':
            sep_pos = text.rfind(sep, offset, chunk_end)
            if sep_pos != -1:
                chunk_end = sep_pos + 1
                break

        chunks.append(text[offset:chunk_end])
        offset = chunk_end

    # add last chunk
    chunks.append(text[offset:])

    return chunks


def load_tacotron(checkpoint_path: str, sampling_rate: int):
    """Loads a Tacotron 2 model."""
    hparams = create_hparams()
    hparams.sampling_rate = sampling_rate

    # load model from a checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()

    return model


def load_waveglow(checkpoint_path: str):
    """Loads a WaveGlow model."""
    waveglow = torch.load(checkpoint_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow


def merge_chunks(audio_paths, output_file_path):
    ffmpeg_cmd = 'ffmpeg'
    if which(ffmpeg_cmd) is None:
        raise ValueError('%s is not installed.' % ffmpeg_cmd)

    # build the command
    command_args = [ffmpeg_cmd, '-y']
    for audio_path in audio_paths:
        command_args += ['-i', audio_path]

    num_files = len(audio_paths)
    ffmpeg_filter = ''.join('[%d:0]' % i for i in range(num_files))
    command_args += [
        '-filter_complex', '%sconcat=n=%d:v=0:a=1[out]' % (ffmpeg_filter, num_files),
        '-map', '[out]',
        output_file_path,
    ]

    subprocess.run(command_args)


def main():
    book_path = 'data/book1.txt'
    prefixes_to_fix = ['ultra']
    chunk_limit = 140
    tacotron_checkpoint_path = '/workspace/project/models/tacotron2_statedict.pt'
    waveglow_checkpoint_path = '/workspace/project/models/waveglow_256channels_converted.pt'
    sampling_rate = 22050
    tmp_chunks_dir = 'data/_tmp'
    output_file_path = 'data/book1.mp3'

    convert_book(book_path, prefixes_to_fix, chunk_limit, tacotron_checkpoint_path,  waveglow_checkpoint_path,
                 sampling_rate, tmp_chunks_dir, output_file_path)


if __name__ == '__main__':
    sys.path.append('waveglow')
    logging.basicConfig(level=logging.DEBUG)
    main()
