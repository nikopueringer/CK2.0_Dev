#!/usr/bin/env -S /home/corridor/CK2.0/corridorkey_v2_runtime/venv/bin/python3
import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.GREEN}=== {title} ==={Colors.RESET}")

def print_subheader(title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- {title} ---{Colors.RESET}")

def clean_path(path_str):
    path_str = path_str.strip()
    # Remove surrounding single/double quotes added by drag-and-drop
    if (path_str.startswith('"') and path_str.endswith('"')) or (path_str.startswith("'") and path_str.endswith("'")):
        path_str = path_str[1:-1]
    # Replace escaped spaces
    path_str = path_str.replace("\\ ", " ")
    path_str = os.path.expanduser(path_str)
    return os.path.abspath(path_str)

def translate_windows_path(path_str):
    cleaned = path_str.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1]
        
    mappings = {
        "S:\\": "/mnt/ssd-storage",
        "S:/": "/mnt/ssd-storage",
        "V:\\": "/mnt/ssd-storage",
        "V:/": "/mnt/ssd-storage",
        "\\\\10.10.10.6\\ssd-storage": "/mnt/ssd-storage",
        "//10.10.10.6/ssd-storage": "/mnt/ssd-storage",
    }
    
    for win_prefix, linux_prefix in mappings.items():
        if cleaned.lower().startswith(win_prefix.lower()):
            rel_path = cleaned[len(win_prefix):]
            linux_path = os.path.join(linux_prefix, rel_path).replace("\\", "/")
            linux_path = "/" + linux_path.lstrip("/")
            return linux_path
            
    if cleaned.startswith("\\\\"):
        return cleaned.replace("\\", "/")
        
    return path_str

def find_alpha_hint(input_path):
    input_path = Path(input_path)
    parent_dir = input_path.parent
    base_name = input_path.stem
    
    # Common video and image extensions
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.png', '.jpg', '.jpeg', '.exr'}
    
    hint_files = []
    if parent_dir.exists():
        for item in parent_dir.iterdir():
            if item.is_file() and item.suffix.lower() in allowed_extensions:
                name_lower = item.name.lower()
                prefix_lower = f"{base_name.lower()}_alphahint"
                if name_lower.startswith(prefix_lower):
                    hint_files.append(item)
                    
    return hint_files[0] if hint_files else None

def find_first_frame_hint(input_path):
    input_path = Path(input_path)
    parent_dir = input_path.parent
    base_name = input_path.stem
    
    # Common image extensions
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.exr'}
    
    hint_files = []
    if parent_dir.exists():
        for item in parent_dir.iterdir():
            if item.is_file() and item.suffix.lower() in allowed_extensions:
                name_lower = item.name.lower()
                prefix_lower = f"{base_name.lower()}_alphahint"
                if name_lower.startswith(prefix_lower):
                    hint_files.append(item)
                    
    return hint_files[0] if hint_files else None

def get_batch_clips(dir_path):
    dir_path = Path(dir_path)
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    
    clips = []
    try:
        # Find video files directly in the directory (excluding hints)
        for item in sorted(dir_path.iterdir()):
            if item.is_file() and item.suffix.lower() in video_extensions:
                if "_alphahint" not in item.name.lower():
                    clips.append(item)
    except Exception as e:
        print(f"{Colors.YELLOW}[WARNING] Error scanning directory: {e}{Colors.RESET}")
        
    return clips

def prompt_choices(title, options, default=1):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    for idx, opt in enumerate(options, 1):
        suffix = " (Default)" if idx == default else ""
        print(f"  {idx}) {opt}{suffix}")
    while True:
        try:
            choice = input(f"Select choice [1-{len(options)}]: ").strip()
            if not choice:
                return default
            val = int(choice)
            if 1 <= val <= len(options):
                return val
        except ValueError:
            pass
        print(f"Invalid selection. Please choose a number between 1 and {len(options)}.")

def prompt_confirm(prompt_str, default=True):
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"\n{Colors.BOLD}{Colors.BLUE}{prompt_str}{Colors.RESET}{suffix}: ").strip().lower()
    if not val:
        return default
    return val.startswith('y')

def prompt_input(prompt_str, default=None, validator=None):
    suffix = f" (Default: {default})" if default is not None else ""
    while True:
        val = input(f"\n{Colors.BOLD}{Colors.BLUE}{prompt_str}{Colors.RESET}{suffix}: ").strip()
        if not val:
            if default is not None:
                return default
            print("This setting is required.")
            continue
        if validator:
            try:
                cleaned = validator(val)
                if cleaned is not None:
                    return cleaned
            except Exception:
                pass
            print(f"{Colors.YELLOW}Invalid input value.{Colors.RESET}")
            continue
        return val

def main():
    print_header("CorridorKey v2 Interactive Runner")
    
    # 1. Resolve Input Path
    input_path_str = ""
    if len(sys.argv) > 1:
        input_path_str = sys.argv[1]
        print(f"Input path passed via argument: {input_path_str}")
    else:
        input_path_str = input(f"{Colors.BOLD}{Colors.BLUE}Drag and drop your video file or directory here and press Enter:{Colors.RESET}\n").strip()
    
    input_path = clean_path(translate_windows_path(input_path_str))
    if not os.path.exists(input_path):
        print(f"{Colors.YELLOW}[ERROR] Resolved Linux input path does not exist: {input_path}{Colors.RESET}")
        sys.exit(1)
        
    print(f"Resolved input path: {Colors.GREEN}{input_path}{Colors.RESET}")
    
    is_batch = os.path.isdir(input_path)
    
    if is_batch:
        # ======================================================================
        # BATCH DIRECTORY MODE
        # ======================================================================
        batch_clips = get_batch_clips(input_path)
        if not batch_clips:
            print(f"{Colors.YELLOW}[ERROR] Directory contains no supported video clips or frame subdirectories: {input_path}{Colors.RESET}")
            sys.exit(1)
            
        print(f"Found {Colors.GREEN}{len(batch_clips)}{Colors.RESET} clips to process in batch.")
        
        # Analyze if any clips have companion hints
        any_have_hints = False
        has_video_hint = False
        for clip in batch_clips:
            hint = find_alpha_hint(clip)
            if hint:
                any_have_hints = True
                if hint.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
                    has_video_hint = True
                    
        # Config options (Hint variables apply only to clips with hints)
        hint_quality = 0.75
        no_carry_hint = False
        carry_hint_quality = 0.95
        
        if any_have_hints:
            print_subheader("Hint Configuration for Batch Clips")
            q_choice = prompt_choices(
                "Select Hint Trust/Quality (applied to clips with AlphaHints):",
                [
                    "High Quality (0.95 - clean artist-provided matte)",
                    "Medium Quality (0.75 - rough automated hint)",
                    "Custom Value (0.0 to 1.0)"
                ],
                default=2
            )
            if q_choice == 1:
                hint_quality = 0.95
            elif q_choice == 2:
                hint_quality = 0.75
            else:
                hint_quality = prompt_input("Enter custom quality value [0.0 - 1.0]", default="0.75", validator=lambda v: float(v) if 0.0 <= float(v) <= 1.0 else None)
                
            if has_video_hint:
                is_authoritative = prompt_confirm("Should video hints be authoritative on every frame (disabling predicted carry)?", default=False)
                no_carry_hint = is_authoritative
                
            if not no_carry_hint:
                carry_hint_quality = prompt_input(
                    "Enter carry hint quality value [0.0 - 1.0] (trust level for predicted hints)",
                    default="0.95",
                    validator=lambda v: float(v) if 0.0 <= float(v) <= 1.0 else None
                )

        # Output configurations
        output_choice = input(f"\n{Colors.BOLD}{Colors.BLUE}The default setting exports all four outputs: alpha, fg, cutout, and checker. To customize, enter any numbers separated by space or press Enter to keep all outputs:{Colors.RESET}\n"
                              "  1) alpha (predicted matte preview) [Default]\n"
                              "  2) fg (predicted foreground RGB preview) [Default]\n"
                              "  3) cutout (ProRes 4444 QuickTime with alpha and despill) [Default]\n"
                              "  4) checker (checkerboard composite with despill preview) [Default]\n"
                              "Selection: ").strip()
        
        outputs = []
        if not output_choice:
            outputs = ["alpha", "fg", "cutout", "checker"]
        else:
            mapping = {1: "alpha", 2: "fg", 3: "cutout", 4: "checker"}
            parts = output_choice.replace(",", " ").split()
            for p in parts:
                try:
                    num = int(p)
                    if num in mapping:
                        outputs.append(mapping[num])
                except ValueError:
                    pass
            if not outputs:
                outputs = ["alpha", "fg", "cutout", "checker"]
        print(f"Selected outputs: {Colors.GREEN}{', '.join(outputs)}{Colors.RESET}")

        cutout_linear = False
        if "cutout" in outputs:
            print_subheader("Cutout Color Space & Alpha Mode")
            cutout_choice = prompt_choices(
                "Select Cutout Color Space & Alpha Mode:",
                [
                    "Straight sRGB color with linear alpha (Default / Standard compatibility)",
                    "Straight linear color with linear alpha (VFX linear)"
                ],
                default=1
            )
            cutout_linear = (cutout_choice == 2)

        despill_strength = 0.5
        if "cutout" in outputs or "checker" in outputs:
            print_subheader("Despill Configuration")
            val = prompt_input(
                "Enter despill strength [0.0 - 1.0] (0.0 is no despill, 1.0 is full despill)",
                default="0.5",
                validator=lambda v: float(v) if 0.0 <= float(v) <= 1.0 else None
            )
            despill_strength = float(val)

        # VRAM profile
        low_vram = prompt_confirm("Enable Low VRAM mode? (Recommended for 4K video or consumer GPUs)", default=False)
        temp_dir = "/tmp/corridorkey_v2"
        if low_vram:
            custom_temp = prompt_confirm("Use a custom directory for feature caching? (Useful if /tmp has limited space)", default=False)
            if custom_temp:
                temp_dir = clean_path(prompt_input("Enter cache directory path:"))

        # Advanced options
        customize_advanced = prompt_confirm("Customize advanced settings (Hann window sizes, output folder name, frame limits)?", default=False)
        
        hann_chunk = 48
        hann_stride = 24
        output_dir_str = ""
        num_frames = -1
        start_mode = "begin"
        
        if customize_advanced:
            window_choice = prompt_choices(
                "Select Hann Window Presets (Temporal Context):",
                [
                    "High VRAM (48 chunk, 24 stride) - Recommended for standard GPUs",
                    "Medium VRAM (24 chunk, 12 stride) - Lower peak memory",
                    "Low VRAM (8 chunk, 4 stride) - Minimal memory, fast startup",
                    "Max Continuity (80 chunk, 40 stride) - Package default (requires high workstation VRAM)",
                    "Custom Size"
                ],
                default=1
            )
            if window_choice == 1:
                hann_chunk = 48
                hann_stride = 24
            elif window_choice == 2:
                hann_chunk = 24
                hann_stride = 12
            elif window_choice == 3:
                hann_chunk = 8
                hann_stride = 4
            elif window_choice == 4:
                hann_chunk = 80
                hann_stride = 40
            else:
                hann_chunk = int(prompt_input("Enter Hann Chunk (integer):", default="48"))
                hann_stride = int(prompt_input("Enter Hann Stride (integer):", default="24"))
                
            output_dir_str = prompt_input("Enter custom output directory path (leave empty to use default folder):", default="")
                
            limit_choice = prompt_input("Enter frame limit (number of frames, or 'all')", default="all")
            if limit_choice.lower() != "all":
                try:
                    num_frames = int(limit_choice)
                    start_mode_choice = prompt_choices(
                        "Select Start Mode for partial clips:",
                        [
                            "Start from the beginning (begin)",
                            "Take a centered segment (middle)",
                            "Take a deterministic random segment (random_middle)"
                        ],
                        default=1
                    )
                    start_modes = ["begin", "middle", "random_middle"]
                    start_mode = start_modes[start_mode_choice - 1]
                except ValueError:
                    print(f"{Colors.YELLOW}Invalid frame limit choice, using 'all'.{Colors.RESET}")
                    num_frames = -1

        custom_output_parent = None
        if output_dir_str:
            custom_output_parent = clean_path(translate_windows_path(output_dir_str))

        # Print Batch Execution Plan
        print_header(f"Batch Execution Plan ({len(batch_clips)} clips)")
        for idx, clip in enumerate(batch_clips, 1):
            hint = find_alpha_hint(clip)
            if hint:
                hint_type = "Full Hint Video" if hint.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'} else "First-Frame Hint"
                print(f"  {idx}) {Colors.CYAN}{clip.name}{Colors.RESET} -> running in {Colors.GREEN}{hint_type}{Colors.RESET} mode (using {hint.name})")
            else:
                print(f"  {idx}) {Colors.CYAN}{clip.name}{Colors.RESET} -> running in {Colors.YELLOW}No Hint{Colors.RESET} mode")

        confirm_batch = prompt_confirm(f"Do you want to run this batch of {len(batch_clips)} clips now?", default=True)
        if not confirm_batch:
            print("Batch execution canceled.")
            sys.exit(0)

        # Run Batch Sequentially
        package_root = Path(__file__).resolve().parent
        for idx, clip in enumerate(batch_clips, 1):
            print_header(f"Processing Clip {idx} of {len(batch_clips)}: {clip.name}")
            
            # Auto-detect hint type for this specific clip
            clip_hint_video_path = None
            clip_hint_first_frame = None
            clip_inference_mode = 1
            
            hint = find_alpha_hint(clip)
            if hint:
                if hint.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm'}:
                    clip_hint_video_path = str(hint)
                    clip_inference_mode = 3
                    print(f"Applying companion AlphaHint video: {Colors.CYAN}{hint.name}{Colors.RESET}")
                else:
                    clip_hint_first_frame = str(hint)
                    clip_inference_mode = 2
                    print(f"Applying companion First-Frame Hint image: {Colors.CYAN}{hint.name}{Colors.RESET}")
            else:
                print(f"Running clip in {Colors.YELLOW}No Hint{Colors.RESET} mode.")

            # Resolve output directory
            if custom_output_parent:
                output_dir = str(Path(custom_output_parent) / f"out_{clip.stem}").replace("\\", "/")
            else:
                output_dir = str(clip.parent / f"out_{clip.stem}").replace("\\", "/")
            print(f"Output directory: {Colors.GREEN}{output_dir}{Colors.RESET}")

            # Assemble command
            cmd = [sys.executable, str(package_root / "infer.py")]
            cmd.extend(["--input", str(clip)])
            cmd.extend(["--output_dir", output_dir])
            
            if clip_hint_first_frame:
                cmd.extend(["--hint_first_frame", clip_hint_first_frame])
            if clip_hint_video_path:
                cmd.extend(["--hint_video", clip_hint_video_path])
                
            if clip_inference_mode in (2, 3) and (clip_hint_first_frame or clip_hint_video_path):
                cmd.extend(["--hint_quality", str(hint_quality)])
                if no_carry_hint:
                    cmd.append("--no_carry_hint")
                else:
                    cmd.extend(["--carry_hint_quality", str(carry_hint_quality)])
                    
            cmd.extend(["--outputs"] + outputs)
            if cutout_linear:
                cmd.append("--cutout_linear")
            cmd.extend(["--despill_strength", str(despill_strength)])
            
            if low_vram:
                cmd.append("--low_vram")
                cmd.extend(["--temp_dir", temp_dir])
                
            cmd.extend(["--hann_chunk", str(hann_chunk)])
            cmd.extend(["--hann_stride", str(hann_stride)])
            
            if num_frames > 0:
                cmd.extend(["--num_frames", str(num_frames)])
                cmd.extend(["--start_mode", start_mode])

            print(f"Running command: {' '.join(cmd)}\n")
            try:
                # Pre-set offline mode variables
                env = os.environ.copy()
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                for line in process.stdout:
                    print(line, end="")
                process.wait()
                
                if process.returncode == 0:
                    print(f"\n{Colors.BOLD}{Colors.GREEN}[SUCCESS] Finished {clip.name} successfully!{Colors.RESET}")
                else:
                    print(f"\n{Colors.BOLD}{Colors.YELLOW}[ERROR] Failed {clip.name} with exit code {process.returncode}{Colors.RESET}")
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}[INTERRUPTED] Batch paused by user.{Colors.RESET}")
                confirm_continue = prompt_confirm("Do you want to continue to the next clip in the batch?", default=True)
                if not confirm_continue:
                    print("Batch execution aborted.")
                    break
        print_header("Batch Directory Processing Complete")
        sys.exit(0)

    else:
        # ======================================================================
        # SINGLE CLIP MODE (with loop capability)
        # ======================================================================
        # 2. Check for Alpha Hint file in the same directory (first run)
        hint_video_path = None
        hint_first_frame = None
        inference_mode = 1 # 1: No Hint, 2: First-Frame, 3: Full Hint Video
        
        discovered_hint = find_alpha_hint(input_path)
        if discovered_hint:
            print(f"\n{Colors.YELLOW}>>> Found matching AlphaHint file: {discovered_hint.name}{Colors.RESET}")
            use_discovered = prompt_confirm(f"Would you like to use this file as the Full Hint Video?", default=True)
            if use_discovered:
                hint_video_path = str(discovered_hint)
                inference_mode = 3
                print(f"Selected mode: {Colors.GREEN}Full Hint Video (using auto-discovered file){Colors.RESET}")
                
        # 3. Prompt for Inference Mode if not already selected via auto-hint
        if inference_mode == 1 and not hint_video_path:
            mode_choice = prompt_choices(
                "Choose Inference Mode:",
                [
                    "No Hint (Model automatically selects the foreground subject)",
                    "First-Frame Hint (Provide starting matte image/directory)",
                    "Full Hint Video (Provide a matte/roto video/directory for every frame)"
                ],
                default=1
            )
            inference_mode = mode_choice
            
            if inference_mode == 2:
                hint_first_frame_str = prompt_input("Enter path to first-frame hint image/matte:")
                hint_first_frame = clean_path(translate_windows_path(hint_first_frame_str))
                if not os.path.exists(hint_first_frame):
                    print(f"{Colors.YELLOW}[ERROR] First frame hint path does not exist: {hint_first_frame}{Colors.RESET}")
                    sys.exit(1)
            elif inference_mode == 3:
                hint_video_str = prompt_input("Enter path to full hint video or directory:")
                hint_video_path = clean_path(translate_windows_path(hint_video_str))
                if not os.path.exists(hint_video_path):
                    print(f"{Colors.YELLOW}[ERROR] Hint video path does not exist: {hint_video_path}{Colors.RESET}")
                    sys.exit(1)

        # 4. Prompt for carry-hint details
        no_carry_hint = False
        hint_quality = 0.75
        carry_hint_quality = 0.95
        
        if inference_mode in (2, 3):
            # Ask for hint quality
            q_choice = prompt_choices(
                "Select Hint Trust/Quality:",
                [
                    "High Quality (0.95 - clean artist-provided matte)",
                    "Medium Quality (0.75 - rough automated hint)",
                    "Custom Value (0.0 to 1.0)"
                ],
                default=2
            )
            if q_choice == 1:
                hint_quality = 0.95
            elif q_choice == 2:
                hint_quality = 0.75
            else:
                def val_quality(v):
                    try:
                        f = float(v)
                        if 0.0 <= f <= 1.0:
                            return f
                    except ValueError:
                        pass
                    print("Please enter a float value between 0.0 and 1.0.")
                    return None
                hint_quality = prompt_input("Enter custom quality value [0.0 - 1.0]", default="0.75", validator=val_quality)
                
            if inference_mode == 3:
                # Full hint video has carry options
                is_authoritative = prompt_confirm("Should the hint video be authoritative on every frame (disabling predicted carry hints)?", default=False)
                no_carry_hint = is_authoritative
                
            if not no_carry_hint:
                carry_q = prompt_input(
                    "Enter carry hint quality value [0.0 - 1.0] (trust level for predicted hints)",
                    default="0.95",
                    validator=lambda v: float(v) if 0.0 <= float(v) <= 1.0 else None
                )
                carry_hint_quality = carry_q

        # 5. Output configurations
        output_choice = input(f"\n{Colors.BOLD}{Colors.BLUE}The default setting exports all four outputs: alpha, fg, cutout, and checker. To customize, enter any numbers separated by space or press Enter to keep all outputs:{Colors.RESET}\n"
                              "  1) alpha (predicted matte preview) [Default]\n"
                              "  2) fg (predicted foreground RGB preview) [Default]\n"
                              "  3) cutout (ProRes 4444 QuickTime with alpha and despill) [Default]\n"
                              "  4) checker (checkerboard composite with despill preview) [Default]\n"
                              "Selection: ").strip()
        
        outputs = []
        if not output_choice:
            outputs = ["alpha", "fg", "cutout", "checker"]
        else:
            mapping = {1: "alpha", 2: "fg", 3: "cutout", 4: "checker"}
            parts = output_choice.replace(",", " ").split()
            for p in parts:
                try:
                    num = int(p)
                    if num in mapping:
                        outputs.append(mapping[num])
                except ValueError:
                    pass
            if not outputs:
                outputs = ["alpha", "fg", "cutout", "checker"]
                
        print(f"Selected outputs: {Colors.GREEN}{', '.join(outputs)}{Colors.RESET}")

        cutout_linear = False
        if "cutout" in outputs:
            cutout_choice = prompt_choices(
                "Select Cutout Color Space & Alpha Mode:",
                [
                    "Straight sRGB color with linear alpha (Default / Standard compatibility)",
                    "Straight linear color with linear alpha (VFX linear)"
                ],
                default=1
            )
            cutout_linear = (cutout_choice == 2)

        despill_strength = 0.5
        if "cutout" in outputs or "checker" in outputs:
            print_subheader("Despill Configuration")
            val = prompt_input(
                "Enter despill strength [0.0 - 1.0] (0.0 is no despill, 1.0 is full despill)",
                default="0.5",
                validator=lambda v: float(v) if 0.0 <= float(v) <= 1.0 else None
            )
            despill_strength = float(val)

        # 6. VRAM profile
        low_vram = prompt_confirm("Enable Low VRAM mode? (Recommended for 4K video or consumer GPUs)", default=False)
        temp_dir = "/tmp/corridorkey_v2"
        if low_vram:
            custom_temp = prompt_confirm("Use a custom directory for feature caching? (Useful if /tmp has limited space)", default=False)
            if custom_temp:
                temp_dir = clean_path(prompt_input("Enter cache directory path:"))

        # 7. Advanced options
        customize_advanced = prompt_confirm("Customize advanced settings (Hann window sizes, output folder name, frame limits)?", default=False)
        
        hann_chunk = 48
        hann_stride = 24
        output_dir_str = ""
        num_frames = -1
        start_mode = "begin"
        
        if customize_advanced:
            # Hann windows
            window_choice = prompt_choices(
                "Select Hann Window Presets (Temporal Context):",
                [
                    "High VRAM (48 chunk, 24 stride) - Recommended for standard GPUs",
                    "Medium VRAM (24 chunk, 12 stride) - Lower peak memory",
                    "Low VRAM (8 chunk, 4 stride) - Minimal memory, fast startup",
                    "Max Continuity (80 chunk, 40 stride) - Package default (requires high workstation VRAM)",
                    "Custom Size"
                ],
                default=1
            )
            if window_choice == 1:
                hann_chunk = 48
                hann_stride = 24
            elif window_choice == 2:
                hann_chunk = 24
                hann_stride = 12
            elif window_choice == 3:
                hann_chunk = 8
                hann_stride = 4
            elif window_choice == 4:
                hann_chunk = 80
                hann_stride = 40
            else:
                hann_chunk = int(prompt_input("Enter Hann Chunk (integer):", default="48"))
                hann_stride = int(prompt_input("Enter Hann Stride (integer):", default="24"))
                
            # Output directory
            output_dir_str = prompt_input("Enter custom output directory path (leave empty to use default folder):", default="")
                
            # Frame limits
            limit_choice = prompt_input("Enter frame limit (number of frames, or 'all')", default="all")
            if limit_choice.lower() != "all":
                try:
                    num_frames = int(limit_choice)
                    start_mode_choice = prompt_choices(
                        "Select Start Mode for partial clips:",
                        [
                            "Start from the beginning (begin)",
                            "Take a centered segment (middle)",
                            "Take a deterministic random segment (random_middle)"
                        ],
                        default=1
                    )
                    start_modes = ["begin", "middle", "random_middle"]
                    start_mode = start_modes[start_mode_choice - 1]
                except ValueError:
                    print(f"{Colors.YELLOW}Invalid frame limit choice, using 'all'.{Colors.RESET}")
                    num_frames = -1

        # Keep track of custom output directory parent to maintain clean batch routing
        custom_output_parent = None
        if output_dir_str:
            custom_output_parent = clean_path(translate_windows_path(output_dir_str))

        is_first_run = True

        # Execution Loop for Single Clip Mode
        while True:
            if not is_first_run:
                print_header("Process Another Video")
                next_input_str = input(f"{Colors.BOLD}{Colors.BLUE}Drag and drop another video file here to process with the same settings (or press Enter to exit):{Colors.RESET}\n").strip()
                if not next_input_str:
                    print("\nExiting interactive runner. Goodbye!")
                    break
                    
                input_path = clean_path(translate_windows_path(next_input_str))
                if not os.path.exists(input_path):
                    print(f"{Colors.YELLOW}[ERROR] Resolved Linux input path does not exist: {input_path}{Colors.RESET}\n")
                    continue
                    
                print(f"Resolved input path: {Colors.GREEN}{input_path}{Colors.RESET}")
                
                # Resolve hints for the new video
                hint_video_path = None
                hint_first_frame = None
                
                if inference_mode == 2:
                    # Try to auto-discover first-frame hint in new folder
                    discovered = find_first_frame_hint(input_path)
                    if discovered:
                        print(f"{Colors.YELLOW}>>> Found matching First-Frame Hint: {discovered.name}{Colors.RESET}")
                        hint_first_frame = str(discovered)
                    else:
                        hint_str = input(f"\n{Colors.BOLD}{Colors.BLUE}Enter path to first-frame hint image/matte for this video (or press Enter to skip hint):{Colors.RESET}\n").strip()
                        if hint_str:
                            hint_first_frame = clean_path(translate_windows_path(hint_str))
                            if not os.path.exists(hint_first_frame):
                                print(f"{Colors.YELLOW}[ERROR] Hint path does not exist. Running this clip without hints.{Colors.RESET}")
                                hint_first_frame = None
                                
                elif inference_mode == 3:
                    # Try to auto-discover full hint video in new folder
                    discovered = find_alpha_hint(input_path)
                    if discovered:
                        print(f"{Colors.YELLOW}>>> Found matching AlphaHint file: {discovered.name}{Colors.RESET}")
                        hint_video_path = str(discovered)
                    else:
                        hint_str = input(f"\n{Colors.BOLD}{Colors.BLUE}Enter path to full hint video/directory for this video (or press Enter to skip hint):{Colors.RESET}\n").strip()
                        if hint_str:
                            hint_video_path = clean_path(translate_windows_path(hint_str))
                            if not os.path.exists(hint_video_path):
                                print(f"{Colors.YELLOW}[ERROR] Hint path does not exist. Running this clip without hints.{Colors.RESET}")
                                hint_video_path = None

            is_first_run = False

            # Determine output directory for the current clip
            if custom_output_parent:
                input_path_obj = Path(input_path)
                output_dir = str(Path(custom_output_parent) / f"out_{input_path_obj.stem}").replace("\\", "/")
            else:
                input_path_obj = Path(input_path)
                output_dir = str(input_path_obj.parent / f"out_{input_path_obj.stem}").replace("\\", "/")
                
            print(f"Output directory: {Colors.GREEN}{output_dir}{Colors.RESET}")

            # Assemble inference command
            package_root = Path(__file__).resolve().parent
            cmd = [sys.executable, str(package_root / "infer.py")]
            
            cmd.extend(["--input", input_path])
            cmd.extend(["--output_dir", output_dir])
            
            if hint_first_frame:
                cmd.extend(["--hint_first_frame", hint_first_frame])
            if hint_video_path:
                cmd.extend(["--hint_video", hint_video_path])
                
            if inference_mode in (2, 3) and (hint_first_frame or hint_video_path):
                cmd.extend(["--hint_quality", str(hint_quality)])
                if no_carry_hint:
                    cmd.append("--no_carry_hint")
                else:
                    cmd.extend(["--carry_hint_quality", str(carry_hint_quality)])
                    
            cmd.extend(["--outputs"] + outputs)
            if cutout_linear:
                cmd.append("--cutout_linear")
            cmd.extend(["--despill_strength", str(despill_strength)])
            
            if low_vram:
                cmd.append("--low_vram")
                cmd.extend(["--temp_dir", temp_dir])
                
            cmd.extend(["--hann_chunk", str(hann_chunk)])
            cmd.extend(["--hann_stride", str(hann_stride)])
            
            if num_frames > 0:
                cmd.extend(["--num_frames", str(num_frames)])
                cmd.extend(["--start_mode", start_mode])

            # Execute
            print_header("Executing Inference Command")
            print(f"Command:\n{Colors.BOLD}{Colors.YELLOW}{' '.join(cmd)}{Colors.RESET}\n")
            
            confirm_run = prompt_confirm("Do you want to run this command now?", default=True)
            if not confirm_run:
                print("Skipping this video.\n")
                continue
                
            try:
                # Pre-set offline mode variables
                env = os.environ.copy()
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                for line in process.stdout:
                    print(line, end="")
                process.wait()
                
                if process.returncode == 0:
                    print(f"\n{Colors.BOLD}{Colors.GREEN}[SUCCESS] CorridorKey v2 inference completed successfully!{Colors.RESET}")
                    print(f"Outputs written to: {Colors.CYAN}{output_dir}{Colors.RESET}\n")
                else:
                    print(f"\n{Colors.BOLD}{Colors.YELLOW}[ERROR] Inference failed with exit code {process.returncode}{Colors.RESET}\n")
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}[INTERRUPTED] Inference run canceled by user.{Colors.RESET}\n")

if __name__ == "__main__":
    main()
