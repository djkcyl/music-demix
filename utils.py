import torch
import numpy
from tqdm import tqdm


def bsr_demix_track(model, mix, device, overlap=2):
    C = 352800
    N = overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = 4

    length_init = mix.shape[-1]

    if length_init > 2 * border and (border > 0):
        mix = torch.nn.functional.pad(mix, (border, border), mode="reflect")

    window_size = C
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)
    window_start[-fade_size:] *= fadeout
    window_finish[:fade_size] *= fadein
    window_middle[-fade_size:] *= fadeout
    window_middle[:fade_size] *= fadein

    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            req_shape = (1,) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []

            # 初始化 tqdm 进度条
            total_steps = (mix.shape[1] + step - 1) // step
            progress_bar = tqdm(
                total=total_steps, desc="Demixing Progress", unit="chunk"
            )

            while i < mix.shape[1]:
                part = mix[:, i : i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = torch.nn.functional.pad(
                            input=part, pad=(0, C - length), mode="reflect"
                        )
                    else:
                        part = torch.nn.functional.pad(
                            input=part,
                            pad=(0, C - length, 0, 0),
                            mode="constant",
                            value=0,
                        )
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step
                progress_bar.update(1)  # 更新进度条

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = window_middle
                    if i - step == 0:  # First audio chunk, no fadein
                        window = window_start
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window = window_finish

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start : start + l] += (
                            x[j][..., :l].cpu() * window[..., :l]
                        )
                        counter[..., start : start + l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

            progress_bar.close()  # 关闭进度条

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            numpy.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    return estimated_sources[0]
