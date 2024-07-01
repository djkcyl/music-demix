import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import librosa  # noqa: E402
import soundfile  # noqa: E402
import numpy  # noqa: E402
from numpy import ndarray  # noqa: E402
from tqdm import tqdm  # noqa: E402
from pathlib import Path  # noqa: E402

from utils import bsr_demix_track# noqa: E402
from models.bs_roformer import bsr_model  # noqa: E402

audio_input_path = Path("in")
audio_output_path = Path("out")
audio_input_path.mkdir(exist_ok=True)
audio_output_path.mkdir(exist_ok=True)

torch.backends.cudnn.benchmark = True

bsr_model_path = Path("model_bs_roformer_ep_317_sdr_12.9755.ckpt")


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

print(f"Model loaded from {bsr_model_path}")
print(f"Model device: {next(bsr_model.parameters()).device}")

for audio_file in audio_input_path.glob("**/*.*"):
    try:
        mix, sr = librosa.load(audio_file, sr=44100, mono=False)
        mix = mix.T
    except Exception as e:
        print(f"Error reading {audio_file}: {e}")
        continue
    input_audios.append((mix, sr, audio_file))
if not input_audios:
    print(f"No audio files found in {audio_input_path}")
    exit()


def run():
    for mix, sr, audio_file in tqdm(input_audios):
        audio_name = audio_file.stem
        if len(mix.shape) == 1:
            mix = numpy.stack([mix, mix], axis=-1)
        mixture = torch.tensor(mix.T, dtype=torch.float32)
        tqdm.write(f"Demixing {audio_file.name}")
        result = bsr_demix_track(model=bsr_model, mix=mixture, device=device, overlap=4).T
        soundfile.write(
            audio_output_path.joinpath(f"{audio_name}_vocals.flac"),
            result,
            samplerate=sr,
            subtype="PCM_24",
        )
        # 将vocals和原曲反相相加，得到伴奏
        torch.cuda.empty_cache()
        soundfile.write(
            audio_output_path.joinpath(f"{audio_name}_accompaniment.flac"),
            (mix - result),
            samplerate=sr,
            subtype="PCM_24",
        )
        tqdm.write(f"Demixed {audio_file.name}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("Interrupted")
