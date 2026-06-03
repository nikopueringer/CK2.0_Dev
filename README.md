# CorridorKey v2 Runtime

Native-resolution video inference package for CorridorKey v2.

This package contains the CorridorKey model code, the recommended v2 inference
weights, and the local foundation-model assets used by the model:

- C-RADIOv4-SO400M
- MoGe-2 ViT-B
- RVM small

Everything loads from package-local paths. The runtime sets Hugging Face offline
mode internally and should not try to download model weights.

## Environment

Use an existing CUDA-capable PyTorch environment. This package was tested with:

```
torch==2.9.1
torchvision==0.24.1
CUDA runtime available
```

Install the non-Torch dependencies from the package root:

```
pip install -r requirements.txt
```

`ffmpeg` is required. The script raises a clear error if `ffmpeg` is missing.

## Quick Start

From inside `corridorkey_v2_runtime/`:

```
python infer.py \
  --input /path/to/input.mp4 \
  --output_dir /path/to/output
```

This runs native-resolution inference with no external hint. By default it
writes:

- `alpha`: predicted matte as a video preview.
- `fg`: predicted foreground RGB as a video preview.

Native resolution is always preserved. The model reflect-pads internally to its
required shape quantum and crops back to the original size for output.

## Choosing An Inference Mode

CorridorKey can run with no hints, a single first-frame hint, or a full hint
video. The right mode depends on what information you have and how much you want
the model to decide for itself. More modes are also possible if we want to add them (for instance, hints for more than one or all frames in the first Hann window and then carried hints after that)

### No Hint

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_no_hint
```

Use this when no matte or object hint is available. The model decides what the
foreground subject is.

Tradeoffs:

- Fastest setup because no hint asset is needed.
- Useful for quick tests and simple shots.
- Less controlled when multiple plausible subjects are present.
- The model may make a different subject-selection decision than you intended.

### First-Frame Hint

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_frame1_hint \
  --hint_first_frame frame_0001_matte.png \
  --hint_quality 0.95
```

Use this when you can provide a good matte for the first frame. The hint should
be a single-channel matte or an image with an alpha channel: white means subject,
black means background.

By default, Hann carry is enabled. This means the model uses its own predicted
auxiliary matte from overlapping Hann windows as a continuing hint signal after
the first explicit hint.

Tradeoffs:

- Usually the best practical mode when only a starting matte is available.
- Gives the model a clear object identity at the beginning of the shot.
- Can track through time without requiring a full hint video.
- Can drift if the shot is long, crowded, or has large occlusions.
- Larger Hann windows can improve continuity but cost more VRAM.

### Full Hint Video

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_full_hint \
  --hint_video hint_matte_video.mp4 \
  --hint_quality 0.95
```

Use this when you have a matte, roto pass, garbage matte, or other subject hint
for every frame.

Tradeoffs:

- Most controlled mode.
- Best when an external tool or artist pass already knows the intended subject.
- The output will tend to follow systematic bias in the hint, so a hint that is
  too eroded, too dilated, or attached to the wrong subject can steer the result
  in that direction.
- If you want the full hint video to be used directly on every pass, add
  `--no_carry_hint`. Otherwise, carried model hints are used on overlapping
  frames while explicit hints are still used for non-carried frames.

### Full Hint Video Without Carry

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_full_hint_no_carry \
  --hint_video hint_matte_video.mp4 \
  --hint_quality 0.95 \
  --no_carry_hint
```

Use this when the hint video is authoritative and you do not want the model's
previous-window predictions to feed back as hints.

Tradeoffs:

- More literal use of the provided hint video.
- Avoids feedback from an imperfect carried prediction.
- Gives up one of the temporal stabilization mechanisms used by the default
  setup.

## Hint Quality

`--hint_quality` tells the model how much to trust an explicit hint. It does not
change the hint image itself; it changes the conditioning value supplied to the
model.

Practical values:

- `0.95`: high-quality matte or artist-provided hint.
- `0.75`: rough but useful automated hint.
- `0.50` or below: loose hint, weak scribble-like mask, or unreliable source.

For carried Hann hints, use `--carry_hint_quality`. The default is `0.95`.

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_frame1_hint \
  --hint_first_frame frame_0001_matte.png \
  --hint_quality 0.95 \
  --carry_hint_quality 0.95
```

If a Hann window has no hints at all, the model receives the no-hint signal and
is asked to decide the subject. If a window has at least one hinted frame,
unhinted frames in that same window receive quality zero, meaning "there is a
hinted subject in this window, but this frame does not have its own explicit
hint."

## Hann Windows

Stage 1 runs on overlapping temporal windows and combines its hidden states with
a Hann-style taper. Stage 2 then streams the final full-resolution predictions
from those combined hidden states.

Default:

```
--hann_chunk 80 --hann_stride 40
```

What the settings mean:

- `hann_chunk`: number of frames processed together by stage 1.
- `hann_stride`: distance between consecutive stage-1 windows.
- `80/40`: 50 percent overlap, high-context default for a high-memory
  workstation GPU.
- `8/4`: lower VRAM and shorter temporal context, sometimes useful for memory
  pressure or quick testing.
- `48/24` or `24/12`: smaller alternatives if `80/40` is too heavy on a
  particular machine.
- Larger windows: often better temporal continuity, higher peak VRAM, more
  latency.
- Smaller windows: lower peak VRAM, weaker long-range temporal consistency.
- `96/48` was not stable at native 4K on the tested 96 GB card; `80/40`
  completed but used most of the available VRAM.

Example:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_window_8 \
  --hann_chunk 8 \
  --hann_stride 4
```

## Outputs

Default outputs are `alpha` and `fg`.

Available output choices:

- `alpha`: matte preview.
- `fg`: raw foreground RGB prediction preview.
- `checker`: predicted foreground composited over a checkerboard.
- `cutout`: predicted foreground multiplied by predicted alpha.
- `despill`: checker composite with the included 0.5-strength despill preview.

`raw_fg` and `foreground` are accepted aliases for `fg`.

Select outputs explicitly:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_review \
  --outputs alpha fg checker cutout despill
```

Performance note: each additional output requires more GPU rendering work and
more ffmpeg encoding. For fastest normal use, keep the default `alpha fg`. Add
`checker`, `cutout`, or `despill` for review renders.

`--stage2_batch` controls how many final-resolution frames are run and rendered
together in stage 2. The default is `4`.

- Higher values can improve throughput if there is enough VRAM.
- Lower values can reduce stage-2 memory pressure.
- At 4K, stage 1 is usually the practical peak-memory path, so Hann window size
  often matters more than `--stage2_batch`.

## Video Encoding

Outputs are written through ffmpeg. Defaults:

```
--ffmpeg_codec libx264 --crf 12 --preset medium --pix_fmt yuv444p
```

Override these as needed for your pipeline. For example:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_encoded \
  --ffmpeg_codec libx264 \
  --crf 10 \
  --preset slow \
  --pix_fmt yuv444p
```

## Long Clips And Low VRAM

The normal path streams video and does not intentionally hold an entire long
clip in RAM. For lower GPU memory, add:

```
python infer.py \
  --input long_shot.mp4 \
  --output_dir out_low_vram \
  --low_vram \
  --temp_dir /path/with/free/space
```

Low-VRAM mode is exact staged inference:

- RVM still runs chronologically through the clip with continuous recurrent
  state.
- C-RADIO and MoGe features are precomputed to temporary memmaps.
- Stage 1 runs Hann windows.
- Stage 2 streams final frames.

Tradeoffs:

- Lower peak VRAM.
- More disk I/O and temporary storage.
- Usually slower.
- Use a temp directory with enough free space for feature memmaps.

Advanced memory knobs:

- `--low_vram_foundation_chunk`: number of frames per VFM precompute chunk.
- `--low_vram_enc_frame_chunk`: stage-1 encoder frame chunking.
- `--low_vram_mlp_chunk_tokens`: MLP token chunking.
- `--low_vram_swin_window_batch`: Swin/window processing chunking.
- `--low_vram_full_attn_query_chunk_tokens`: query chunking for full attention.

Use the defaults unless you are actively tuning memory behavior.

## Frame Directories

`--input` can be a video file or a directory of frames. Frame directories are
read in sorted filename order. Supported extensions are EXR, PNG, JPG, and JPEG.

```
python infer.py \
  --input /path/to/frames \
  --frame_dir_fps 24 \
  --output_dir out_frames
```

EXR frame directories are treated as display-coded by default. If the EXRs are
scene-linear RGB, pass:

```
--frame_dir_exr_linear
```

Example:

```
python infer.py \
  --input /path/to/linear_exr_frames \
  --frame_dir_fps 24 \
  --frame_dir_exr_linear \
  --output_dir out_exr
```

## Partial Clips

For quick tests, process only part of a clip:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_test \
  --num_frames 100 \
  --start_mode middle
```

`--start_mode` choices:

- `begin`: start from the first frame.
- `middle`: take a centered segment.
- `random_middle`: choose a deterministic random middle segment using `--seed`.

## Input Color Handling

The runtime does not inspect or convert color metadata. It uses a simple,
explicit input contract:

- Normal video files are decoded with OpenCV/ffmpeg into RGB values in `[0, 1]`
  and treated as display-coded sRGB/Rec.709-like footage.
- PNG/JPG frame directories are treated the same way: display-coded RGB in
  `[0, 1]`.
- EXR frame directories are also treated as display-coded by default.
- If EXR frames are already scene-linear RGB, pass `--frame_dir_exr_linear`.

For display-coded inputs, the runtime converts RGB to an internal scene-linear
working value with the package gamma function, then applies the model's signed
asinh HDR encoding. Model outputs are decoded back through the inverse path for
video previews.

For `--frame_dir_exr_linear`, the EXR RGB values are passed as scene-linear
values directly into the model's signed asinh encoding. This is the path to use
for unclipped linear EXR workflows.

The foundation models are different: C-RADIO, MoGe, and RVM receive SDR-style
RGB proxies because those models expect normal image-like inputs. They do not
receive unclipped HDR values.

What this runtime does not currently do:

- It does not automatically convert ACEScg, Rec.2020, LogC, S-Log, V-Log, PQ,
  HLG, or camera-raw color spaces.
- It does not use video color metadata to choose a transfer function or primary
  matrix.
- It does not preserve HDR values from ordinary encoded video containers unless
  those values have already been converted into a supported frame input path.

Practical guidance:

- For ordinary sRGB/Rec.709-looking MP4/ProRes review footage, use the video
  directly.
- For linear EXR plates, use a frame directory and pass
  `--frame_dir_exr_linear`.
- For Log, PQ/HLG, ACEScg, Rec.2020, or other managed-color sources, convert
  upstream into either display-coded RGB for the video path or scene-linear RGB
  EXR frames for the linear EXR path.

## Brightness And Contrast Controls

Leave these controls at default for standard inference:

```
--linear_brightness 1.0
--linear_contrast 1.0
--linear_contrast_pivot 0.18
```

The brightness and contrast options apply an experimental linear-light adjustment
after display-coded inputs have been converted to linear light and before the
model's asinh encoding. The inverse adjustment is applied to the predicted
foreground before writing preview outputs. These controls are useful for
controlled tests, not for ordinary production use.

## Failure Behavior

The runtime intentionally does not silently switch modes. Missing weights,
missing foundation assets, missing ffmpeg, unsupported inputs, and load
mismatches raise errors instead of falling back to a different behavior.

## Common Recipes

No hint, fastest standard outputs:

```
python infer.py --input shot.mp4 --output_dir out_no_hint
```

First-frame hint with default carry:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_frame1 \
  --hint_first_frame matte_frame_0001.png \
  --hint_quality 0.95
```

Full hint video, trusting the provided hint on every pass:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_full_hint \
  --hint_video hint.mp4 \
  --hint_quality 0.95 \
  --no_carry_hint
```

Review render with checker and despill:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_review \
  --hint_first_frame matte_frame_0001.png \
  --outputs alpha fg checker despill
```

Lower-memory 8-frame windows:

```
python infer.py \
  --input shot.mp4 \
  --output_dir out_lowmem_window \
  --hint_first_frame matte_frame_0001.png \
  --hann_chunk 8 \
  --hann_stride 4
```
