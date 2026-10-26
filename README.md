<div align="center">

# Speech Enhancement with a 1-D U-Net

### Strip background noise out of a recording, then show what it did to the transcript.

A waveform-domain denoiser trained on real speech and real environmental noise, wrapped in a browser interface that transcribes the audio before and after so you can hear and read the difference.

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white">
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white">
<img alt="gradio" src="https://img.shields.io/badge/UI-Gradio-f97316?style=flat-square">
<img alt="whisper" src="https://img.shields.io/badge/ASR-Whisper%20base-412991?style=flat-square">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square">
</p>

</div>

---

## What it does

Upload a noisy recording. The model runs it through a 1-D U-Net that operates
directly on the waveform, and the app gives you back four things: the cleaned
audio, a before-and-after spectrogram, a transcript of the original, and a
transcript of the cleaned version.

The two transcripts are the interesting part. A denoiser that looks good on a
spectrogram can still destroy the words, and comparing what Whisper hears
before and after makes that visible instead of leaving it to guesswork.

```bash
python app.py        # opens a local Gradio interface
```

---

## The model

Noise reduction happens in the time domain. There is no STFT and no phase
reconstruction, so the output cannot pick up the metallic artefacts that come
from rebuilding a signal out of a modified magnitude spectrogram.

```
input   32,000 samples (2 s @ 16 kHz)
  encoder    Conv1d  1 ->  32   stride 4     8,000
             Conv1d 32 ->  64   stride 4     2,000
             Conv1d 64 -> 128   stride 4       500
  bottleneck Conv1d 128 -> 128                 500
  decoder    ConvT 128 ->  64   + skip e2    2,000
             ConvT 128 ->  32   + skip e1    8,000
             ConvT  64 ->   1              32,000
output  32,000 samples
```

| | |
|---|---|
| parameters | **230,593** |
| checkpoint | 927 KB |
| loss | L1 (mean absolute error) |
| optimiser | Adam, lr 1e-3 |
| sample rate | 16 kHz, mono |

The skip connections are what make this a U-Net rather than a plain
autoencoder. Downsampling by 64× in three steps throws away the fine timing
detail that speech intelligibility depends on, so each decoder stage is handed
the matching encoder output and keeps it.

---

## Getting it running

```bash
git clone https://github.com/rithin415-R/speech-enhancement-unet.git
cd speech-enhancement-unet

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux and macOS

pip install -r requirements.txt
```

Whisper needs **ffmpeg** on the PATH. On Windows, `winget install ffmpeg`.

### Data

The corpora are not in this repository. One command fetches and prepares them:

```bash
python src/download_data.py
```

| | |
|---|---|
| speech | [LibriSpeech](https://www.openslr.org/12) `dev-clean`, 338 MB |
| noise | [ESC-50](https://github.com/karoldvl/ESC-50), 646 MB |

The script mixes each utterance with a random environmental recording and
writes matched pairs into `data/clean` and `data/noisy`, which is what the
training set is: the same utterance with and without noise, so the model has
something exact to aim at.

### Training

```bash
python -m src.train
```

Writes `models/best_model.pth`. The checkpoint in this repository was trained
on **201 matched pairs for 10 epochs** at batch size 2, on CPU. That is a
small run, and the section below says what it does and does not buy you.

---

## What this is, and what it is not

**It is a complete pipeline.** Download, mix, train, infer, serve, transcribe.
Every stage runs, and the trained checkpoint is included so you can try the
app without training anything.

**It is not a benchmarked system.** There is no PESQ or STOI score here, and
`src/evaluate.py` is an empty placeholder. Nothing in this repository
establishes how much this model improves intelligibility, and the honest way
to read the demo is by listening to it and comparing the two transcripts.

**The "Estimated Noise Removed" readout in the app is not a noise measurement.**
It computes an SNR between the enhanced signal and the difference between the
enhanced and the noisy input, which rewards leaving the audio untouched: a
model that passed its input straight through would score highest. Treat that
number as a measure of how much the model changed the signal, not of how much
noise it removed. Replacing it with PESQ and STOI against a held-out split is
the first thing this project needs.

**201 training pairs is small.** Expect it to work best on noise resembling
what it saw, and to generalize poorly to the rest.

---

## Layout

```
app.py                  Gradio interface: enhance, transcribe, plot
requirements.txt
src/
  model.py              the 1-D U-Net
  dataset.py            matched-pair loader, random 2 s crops
  download_data.py      fetch LibriSpeech and ESC-50, mix, write pairs
  train.py              training loop
  evaluate.py           empty, see above
models/
  best_model.pth        the trained checkpoint, 927 KB
```

---

## Licence

MIT, see [`LICENSE`](LICENSE).

LibriSpeech is CC BY 4.0. ESC-50 is CC BY-NC 3.0, which is
**non-commercial**: it may be used to train and evaluate this model, but not
in a commercial product. Whisper is MIT.
