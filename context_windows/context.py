import numpy as np
from typing import Callable, Optional, List
import torch
from ..utils import log
from typing import Callable, Optional, List, Generator

def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)

def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1

def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        # 1) subtract each element by start_val to move vals relative to the start of all frames
        # 2) add num_frames and take modulus to get adjusted vals
        window[i] = ((window[i] - start_val) + num_frames) % num_frames

def shift_window_to_end(window: list[int], num_frames: int):
    # 1) shift window to start
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        # 2) add end_delta to each val to slide windows to end
        window[i] = window[i] + end_delta

def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes

def uniform_looped(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]

#from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
def uniform_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + context_size * context_step, context_step)])

    # now that windows are created, shift any windows that loop, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window is rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]  # roll_val might not be 0 for windows of higher strides
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window (cyclical) is missing roll_val
            if roll_val not in windows[(win_i+1) % len(windows)]:
                # need to insert new window here - just insert window starting at roll_val
                windows.insert(win_i+1, list(range(roll_val, roll_val + context_size)))
        # delete window if it's not unique
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # reverse delete_idxs so that they will be deleted in an order that doesn't break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)
    return windows

def static_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = context_size - context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + context_size
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + context_size)))
            break
        windows.append(list(range(start_idx, start_idx + context_size)))
    return windows
    
def progressive(
    step: int,
    num_steps: Optional[int],
    num_frames: int,
    context_size: Optional[int],
    context_stride: int = 4,
    context_overlap: int = 4,
    **kwargs,
) -> Generator[List[int], None, None]:
    """
    Generates windows in two passes: a coarse pass with large strides for global
    consistency, followed by a fine pass with smaller, overlapping windows for detail.
    """
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    windows = []
    
    # Coarse Pass: large, mostly non-overlapping windows
    delta_coarse = context_size
    for start_idx in range(0, num_frames, delta_coarse):
        ending = start_idx + context_size
        if ending > num_frames:
            start_idx = num_frames - context_size
        windows.append(list(range(start_idx, start_idx + context_size)))

    # Fine Pass: standard overlapping windows
    delta_fine = max(1, context_size - context_overlap)
    for start_idx in range(0, num_frames, delta_fine):
        ending = start_idx + context_size
        if ending >= num_frames:
            final_start_idx = num_frames - context_size
            windows.append(list(range(final_start_idx, final_start_idx + context_size)))
            break
        windows.append(list(range(start_idx, start_idx + context_size)))

    # Remove duplicates, sort, and yield
    unique_windows_tuples = sorted(list(set(tuple(w) for w in windows)))
    for w_tuple in unique_windows_tuples:
        yield list(w_tuple)
        
def meet_in_the_middle(
    step: int,
    num_steps: Optional[int],
    num_frames: int,
    context_size: Optional[int],
    context_stride: int = 4,
    context_overlap: int = 4,
    **kwargs,
) -> Generator[List[int], None, None]:
    """
    Experimental scheduler that generates windows from the start towards the middle,
    and from the end towards the middle, meeting in the center.
    """
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    windows = []
    delta = max(1, context_size - context_overlap)
    midpoint = num_frames // 2

    # Forward pass from the beginning
    start_idx = 0
    while start_idx + context_size <= midpoint + context_size // 2:
        windows.append(list(range(start_idx, start_idx + context_size)))
        start_idx += delta

    # Backward pass from the end
    start_idx = num_frames - context_size
    while start_idx >= midpoint - context_size // 2:
        windows.append(list(range(start_idx, start_idx + context_size)))
        start_idx -= delta
        if start_idx < 0:
            break
            
    # Ensure the very first and last windows are always included
    windows.append(list(range(0, context_size)))
    windows.append(list(range(num_frames - context_size, num_frames)))

    # Clean up, remove duplicates, sort, and yield
    unique_windows_tuples = sorted(list(set(tuple(w) for w in windows)))
    for w_tuple in unique_windows_tuples:
        yield list(w_tuple)     

# START OF CHANGES
def generative_bridge(
    step: int,
    num_steps: Optional[int],
    num_frames: int,
    context_size: Optional[int],
    context_overlap: int,
    **kwargs,
) -> Generator[List[int], None, None]:
    """
    A scheduler that creates two types of windows: 'full' and 'bridge'.
    - 'Full' windows are of `context_size` and are tiled without overlap.
    - 'Bridge' windows are of `2 * context_overlap` length and are generated
      by taking the end of a full window and the start of the next one.
    """
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    # 1. Generate the 'Full' windows (tiled without overlap)
    full_windows = []
    start_idx = 0
    while start_idx < num_frames:
        end_idx = min(start_idx + context_size, num_frames)
        full_windows.append(list(range(start_idx, end_idx)))
        start_idx += context_size

    # 2. Generate the 'Bridge' windows between pairs of full windows
    bridge_windows = []
    if context_overlap > 0:
        for i in range(len(full_windows) - 1):
            prev_window = full_windows[i]
            next_window = full_windows[i+1]

            # Cannot create a valid bridge if windows are too short
            if len(prev_window) < context_overlap or len(next_window) < context_overlap:
                continue

            slice_from_prev = prev_window[-context_overlap:]
            slice_from_next = next_window[:context_overlap]
            bridge_windows.append(slice_from_prev + slice_from_next)

    # 3. Combine, sort by start frame, and yield unique windows
    all_windows = full_windows + bridge_windows
    all_windows.sort(key=lambda w: w[0])

    yielded_windows = set()
    for window in all_windows:
        win_tuple = tuple(window)
        if win_tuple not in yielded_windows:
            yield window
            yielded_windows.add(win_tuple)
# END OF CHANGES        

def get_context_scheduler(name: str) -> Callable:
    """Factory function to select a context scheduler by name."""
    schedulers = {
        "uniform_looped": uniform_looped,
        "uniform_standard": uniform_standard,
        "static_standard": static_standard,
        "progressive": progressive,
        "meet_in_the_middle": meet_in_the_middle,
        "generative_bridge": generative_bridge,
    }
    
    scheduler_func = schedulers.get(name)
    
    if scheduler_func is None:
        raise ValueError(f"Unknown context scheduler named '{name}'.")
        
    return scheduler_func


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )

def create_window_mask(
    noise_pred_context: torch.Tensor, c: List[int], latent_video_length: int, context_overlap: int,
    looped: bool = False, window_type: str = "linear", sigma: float = 0.4, alpha: float = 0.5
) -> torch.Tensor:
    """
    Creates a blending mask for a context window to allow for smooth transitions.
    """

    # If there is no overlap, return a mask of ones immediately
    if context_overlap == 0:
        return torch.ones_like(noise_pred_context)
    
    window_mask = torch.ones_like(noise_pred_context)
    length = noise_pred_context.shape[1]
    device = noise_pred_context.device
    dtype = noise_pred_context.dtype
    
    if window_type == "pyramid":
        # Pyramid (triangular) weights, peaking in the middle of the window.
        if length % 2 == 0:
            max_weight = length // 2
            weight_sequence = list(range(1, max_weight + 1)) + list(range(max_weight, 0, -1))
        else:
            max_weight = (length + 1) // 2
            weight_sequence = list(range(1, max_weight)) + [max_weight] + list(range(max_weight - 1, 0, -1))
        
        weights = torch.tensor(weight_sequence, device=device, dtype=dtype) / max(weight_sequence)
        weights_tensor = weights.view(1, -1, 1, 1)
        window_mask = weights_tensor.expand_as(window_mask).clone()
        
    elif window_type == "gaussian":
        # Gaussian (bell curve) weights.
        if length == 1:
            return torch.ones_like(noise_pred_context)
        n = torch.arange(length, device=device, dtype=dtype)
        center = (length - 1) / 2.0
        sigma_val = sigma * center
        if sigma_val < 1e-6:
            sigma_val = 1e-6
        exponent = -0.5 * torch.pow((n - center) / sigma_val, 2)
        weights = torch.exp(exponent)
        weights_tensor = weights.view(1, -1, 1, 1)
        window_mask = weights_tensor.expand_as(noise_pred_context)

    else:  # Default "linear" window masking (trapezoid shape).
        if min(c) > 0 or (looped and max(c) == latent_video_length - 1):
            ramp_up = torch.linspace(0, 1, context_overlap, device=device, dtype=dtype).view(1, -1, 1, 1)
            window_mask[:, :context_overlap] = ramp_up
            
        if max(c) < latent_video_length - 1 or (looped and min(c) == 0):
            ramp_down = torch.linspace(1, 0, context_overlap, device=device, dtype=dtype).view(1, -1, 1, 1)
            window_mask[:, -context_overlap:] = ramp_down
            
    return window_mask

class WindowTracker:
    def __init__(self, verbose=False):
        self.window_map = {}  # Maps frame sequence to persistent ID
        self.next_id = 0
        self.cache_states = {}  # Maps persistent ID to teacache state
        self.verbose = verbose
    
    def get_window_id(self, frames):
        key = tuple(sorted(frames))  # Order-independent frame sequence
        if key not in self.window_map:
            self.window_map[key] = self.next_id
            if self.verbose:
                log.info(f"New window pattern {key} -> ID {self.next_id}")
            self.next_id += 1
        return self.window_map[key]
    
    def get_teacache(self, window_id, base_state):
        if window_id not in self.cache_states:
            if self.verbose:
                log.info(f"Initializing persistent teacache for window {window_id}")
            self.cache_states[window_id] = base_state.copy()
        return self.cache_states[window_id]
