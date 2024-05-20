import torch
import librosa
import soundfile
import numpy
from numpy import ndarray
from tqdm import tqdm
from pathlib import Path
from models.bs_roformer import BSRoformer

from utils import bsr_demix_track
import warnings

bsr_model_path = Path("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
audio_input_path = Path("in")
audio_output_path = Path("out")
audio_input_path.mkdir(exist_ok=True)
audio_output_path.mkdir(exist_ok=True)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

bsr_model = BSRoformer(
    dim=512,
    depth=12,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    linear_transformer_depth=0,
    freqs_per_bands=tuple(
        [2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128, 129]
    ),
    dim_head=64,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    flash_attn=True,
    dim_freqs_in=1025,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

bsr_state_dict = torch.load(bsr_model_path, map_location=device)
bsr_model.load_state_dict(bsr_state_dict)
bsr_model.to(device)

bsr_model.eval()
input_audios: list[tuple[ndarray, int | float, Path]] = []
instruments = ["vocals"]

print(f"Model loaded from {bsr_model_path}")
print(f"Model device: {next(bsr_model.parameters()).device}")

for audio_file in audio_input_path.glob("**/*.*"):
    try:
        mix, sr = librosa.load(audio_file, sr=44100, mono=False)
        print(f"Loaded {audio_file.name} with shape {mix.shape}, duration {mix.shape[1] / sr:.2f}s")
        mix = mix.T
    except Exception as e:
        print(f"Error reading {audio_file}: {e}")
        continue
    input_audios.append((mix, sr, audio_file))
if not input_audios:
    print(f"No audio files found in {audio_input_path}")
    exit()


for mix, sr, audio_file in tqdm(input_audios):
    if len(mix.shape) == 1:
        mix = numpy.stack([mix, mix], axis=-1)
    mixture = torch.tensor(mix.T, dtype=torch.float32)
    print(f"Demixing {audio_file.name}")
    res = bsr_demix_track(model=bsr_model, mix=mixture, device=device, overlap=2)
    soundfile.write(
        audio_output_path.joinpath(audio_file.stem + "_vocals.flac"),
        res.T,
        samplerate=sr,
        subtype="PCM_24",
    )
    # 将vocals和原曲反相相加，得到伴奏
    soundfile.write(
        audio_output_path.joinpath(audio_file.stem + "_accompaniment.flac"),
        (mix - res.T),
        samplerate=sr,
        subtype="PCM_24",
    )
    tqdm.write(f"Demixed {audio_file.name}")
