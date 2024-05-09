import subprocess
from collections import OrderedDict
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import slider
from slider.curve import Catmull, Linear, MultiBezier, Perfect


class StandardBeatmapRenderer:

    def __init__(self):
        ### time settings
        self.fps = 30
        self.max_frame = None
        self.radius_t = 2
        self.approach_circle_ms = 500
        self.approach_circle_frame = int(self.approach_circle_ms // (1000 // self.fps))
        self.circle_fade_out_ms = 150
        self.circle_fade_out_frame = int(self.circle_fade_out_ms // (1000 // self.fps))

        ### size settings
        self.radius_xy = 7
        self.fade_out_r = 14
        self.approach_circle_r = 28
        self.approach_circle_width = 2
        self.slider_width = 5

        ### other settings
        self.n_mids_curve = 20
        self.n_mids_bezier = 100
        self.wh_scale = 4.0
        self.base_resolution_xy = np.array((512, 384))
        self.padding_xy = np.array((10, 10))
        self.blur_sigma = 2
        self.channels_desc_dict = OrderedDict({
            k: i
            for i, k in enumerate([
                "circle", "slider_head", "slider_body", "slider_tail", "spinner_head",
                "spinner_body", "spinner_tail"
            ])
        })
        self.circle = self._init_hit_circle()
        self.approach_circle = self._init_narrowing_ring()
        self.circle_with_approach = self.mix_shapes_centering([self.circle, self.approach_circle],
                                                              [0, 0])

    def _scale_and_pad(self, x, y):
        """Apply rescale and padding to the hitobject positions.
        """
        if isinstance(x, np.ndarray):
            data = np.stack([x, y], -1)
            data = (data // self.wh_scale).astype(int)
            data = data + self.padding_xy
            return data.T
        else:
            return (int(x // self.wh_scale + self.padding_xy[0]),
                    int(y // self.wh_scale + self.padding_xy[1]))

    def _init_hit_circle(self) -> np.ndarray:
        """Create a hitcircle with fade-in and fade-out."""
        inner_resolution_ratio = 32  # since cv2.circle's radius must be int, upresolution is needed for smooth animation
        r_base = self.radius_xy * inner_resolution_ratio
        r_large = self.fade_out_r * inner_resolution_ratio
        img_shape = np.array(
            (r_large * 2 + inner_resolution_ratio, r_large * 2 + inner_resolution_ratio))
        images = []

        ### fade-in
        for i in range(self.approach_circle_frame):
            img = np.zeros(img_shape, dtype=np.uint8)
            # color changes depending on the frame
            cv2.circle(img, img_shape // 2, r_base, 255 * i // self.approach_circle_frame, -1)
            img = cv2.resize(img, (self.fade_out_r * 2 + 1, self.fade_out_r * 2 + 1),
                             interpolation=cv2.INTER_AREA)
            images.append(img)

        ### fade-out
        for i in range(self.approach_circle_frame,
                       self.approach_circle_frame + self.circle_fade_out_frame):
            # when fading out, r expands into `fade_out_r``. color also changes.
            img = np.zeros(img_shape, dtype=np.uint8)
            r_now = r_base + (r_large - r_base) * (
                i - self.approach_circle_frame) // self.circle_fade_out_frame
            color_now = 255 * (1 - (i - self.approach_circle_frame) / self.circle_fade_out_frame)
            cv2.circle(img, img_shape // 2, r_now, color_now, -1)
            img = cv2.resize(img, (self.fade_out_r * 2 + 1, self.fade_out_r * 2 + 1),
                             interpolation=cv2.INTER_AREA)
            images.append(img)

        assert len(images) == self.approach_circle_frame + self.circle_fade_out_frame
        return np.stack(images)

    def _init_narrowing_ring(self, r_start=None, r_end=None, n_frames=None,
                             inner_resolution_ratio=32, fade=True) -> np.ndarray:
        """Create a 3D array of a ring (circle which is not filled with) that narrows over time.
        inner_resolution_ratio exists since cv2.circle's radius must be int and upresolution is needed for smooth animation"""
        if n_frames is None:
            n_frames = self.approach_circle_frame
        if r_start is None:
            r_start = self.approach_circle_r
        if r_end is None:
            r_end = self.radius_xy
        line_width = self.approach_circle_width
        r_start, r_end, line_width = map(lambda x: x * inner_resolution_ratio,
                                         (r_start, r_end, line_width))

        center = (r_start + line_width, r_start + line_width)
        img_shape = ((r_start + line_width) * 2 + inner_resolution_ratio,
                     (r_start + line_width) * 2 + inner_resolution_ratio)
        images = []
        for i in range(n_frames):
            img = np.zeros(img_shape, dtype=np.uint8)
            color = int(255 * i // n_frames) if fade else 255
            cv2.circle(img, center, r_start - (r_start - r_end) * i // (n_frames - 1), color,
                       line_width)
            img = cv2.resize(img, ((r_start * 2) // inner_resolution_ratio + 1,
                                   (r_start * 2) // inner_resolution_ratio + 1),
                             interpolation=cv2.INTER_AREA)
            images.append(img)
        return np.stack(images)

    def _get_slider_bbox_xyxy(self, midpoints, exclusive=True):
        """Get the bounding box of the slider in the form of (min_x, min_y, max_x, max_y). Max values are exclusive."""
        min_x = midpoints[:, 0].min()
        min_y = midpoints[:, 1].min()
        max_x = midpoints[:, 0].max() + int(exclusive)
        max_y = midpoints[:, 1].max() + int(exclusive)
        return np.array([min_x, min_y, max_x, max_y])

    def _crop_bbox(self, shape, object, small_corner, large_corner):
        """Crop object to fit within the given shape and returns the cropped object and corners."""
        small_corner_org = small_corner.copy()
        large_corner_org = large_corner.copy()
        if np.any(small_corner < 0) or np.any(large_corner > np.array(shape)):
            small_corner = np.clip(small_corner, 0, None)
            large_corner = np.clip(large_corner, None, np.array(shape))
            small_change = small_corner - small_corner_org
            large_change = large_corner - large_corner_org
            object = object[tuple(
                slice(small_change[i], large_corner_org[i] - small_corner_org[i] + large_change[i])
                for i in range(3))]

        return object, small_corner, large_corner

    def _insert_slider(self, base_array, slider, start_idx):
        """Insert `slider` into `base_array` in-place and also returns.
        
        Args:
            base_array (ndarray): The base array to insert the slider into.
                Shape: (T, W, H)
            slider (Slider): The slider object to insert.
            start_idx (int): The starting index in the base array to insert the slider.
        
        Returns:
            ndarray: The modified base array with the slider inserted.
                Shape: (T, W, H)
        """
        assert base_array.ndim == 3, f"base_array.ndim must be 3, but {base_array.ndim}"
        idx_fade_start = start_idx - self.approach_circle_frame
        idx_main_start = start_idx
        idx_main_end = int(idx_main_start + self.timedelta_to_ms(slider.end_time - slider.time) //
                           (1000 / self.fps))
        idx_fade_end = int(idx_main_end + self.circle_fade_out_frame)

        mid_positions = self._scale_and_pad(
            *slider.calc_positions_at(np.linspace(0, 1, self.get_n_mids(slider))).T).T
        bbox_xyxy = self._get_slider_bbox_xyxy(mid_positions).astype(int)
        positionsT = (mid_positions.astype(int) - bbox_xyxy[:2])[:, ::-1]

        ### render slider body
        # slider path
        img = np.zeros((bbox_xyxy[3] - bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0]), dtype=np.uint8)
        cv2.polylines(img, [mid_positions.astype(int) - bbox_xyxy[:2]], False, 255,
                      self.slider_width, lineType=cv2.LINE_AA)
        base_array[idx_main_start:idx_main_end, bbox_xyxy[0]:bbox_xyxy[2], bbox_xyxy[1]:bbox_xyxy[3]] +=  \
            img.T
        # fill the current position of each frame with a circle
        slider_circle_positions = self._scale_and_pad(
            *slider.calc_positions_at(np.linspace(0, 1, idx_main_end -
                                                  idx_main_start)).T).T[:, ::-1]
        for i, t in enumerate(range(idx_main_start, idx_main_end)):
            cv2.circle(base_array[t], slider_circle_positions[i], self.radius_xy, 255, -1)

        ### render slider fade-in
        for t in range(idx_fade_start, idx_main_start):
            color = 255 * (t - idx_fade_start) / (idx_main_start - idx_fade_start)
            cv2.polylines(base_array[t, bbox_xyxy[0]:bbox_xyxy[2], bbox_xyxy[1]:bbox_xyxy[3]],
                          [positionsT], False, color, self.slider_width, lineType=cv2.LINE_AA)

        ### render slider fade-out
        for t in range(idx_main_end, idx_fade_end):
            color = 255 * (1 - (t - idx_main_end) / (idx_fade_end - idx_main_end))
            cv2.polylines(base_array[t, bbox_xyxy[0]:bbox_xyxy[2], bbox_xyxy[1]:bbox_xyxy[3]],
                          [positionsT], False, color, self.slider_width, lineType=cv2.LINE_AA)

        return base_array

    def _insert_object(self, data, object, pos, offset_t=0):
        """Insert `object` into `data` at `pos` in-place and also returns.
        
        Args:
            data (ndarray): The data array to insert the object into.
                Shape: (C, T, W, H)
            object (ndarray): The object array to insert.
                Shape: (T, W, H)
            pos (tuple): The position to insert the object.
                Format: (channel, time, x, y), where x and y are the center of the object.
            offset_t (int, optional): The delay between the object's time and pos.
                Defaults to 0.
        
        Returns:
            ndarray: The modified data array with the object inserted.
                Shape: (C, T, W, H)
        """
        small_corner = np.array((pos[1] + offset_t, *(pos[2:] - np.array(object.shape)[1:] // 2)),
                                dtype=int)
        large_corner = small_corner + np.array(object.shape)

        # check each corners are in the range of data
        object, small_corner, large_corner = self._crop_bbox(data.shape[1:], object, small_corner,
                                                             large_corner)

        mask = object > 0
        data[pos[0], small_corner[0]:large_corner[0], small_corner[1]:large_corner[1],
             small_corner[2]:large_corner[2]][mask] += object[mask]
        return data

    def _insert_spinner(self, data, duration, pos, offset_t=0):
        """
        Insert a spinner into `data` at `pos` in-place and also returns.

        Args:
            data (ndarray): The data array to insert the spinner into.
                Shape: (C, T, W, H)
            duration (int): The duration of the spinner in frames.
            pos (tuple): The position to insert the spinner.
                Format: (channel, time, x, y), where x and y are the center of the spinner.
            offset_t (int, optional): The delay between the spinner's time and pos.
                Defaults to 0.

        Returns:
            ndarray: The modified data array with the spinner inserted.
                Shape: (C, T, W, H)
        """

        r = int(self.base_resolution_xy[0] / 2 / self.wh_scale)
        narroing_ring = self._init_narrowing_ring(r_start=r, r_end=0, n_frames=duration,
                                                  inner_resolution_ratio=1, fade=False)

        # add fade-in whose length is self.approach_circle_frame into narrowing_ring's head
        fade_img = []
        for i in range(self.approach_circle_frame):
            img = np.zeros(narroing_ring[0].shape, dtype=np.uint8)
            cv2.circle(img,
                       np.array(img.shape[:2]) // 2, r, 255 * i // self.approach_circle_frame,
                       self.approach_circle_width)
            fade_img.append(img)
        narroing_ring = np.concatenate([fade_img, narroing_ring], axis=0)

        self._insert_object(data, narroing_ring, pos, offset_t)

    def timedelta_to_ms(self, t, frame_ms=None):
        """Convert `timedelta` to `int` in milliseconds.
        If `frame_ms` is specified, the result is divided by it so returns the frame number."""
        return int(t.total_seconds() * 1000 // (1 if frame_ms is None else frame_ms))

    def render_slider_path_2d(self, slider: slider.Slider, with_background=False) -> np.ndarray:
        """Render a slider path as lines, to a 2D array.
        If `with_background` is `True`, the slider is rendered in a full-size image."""
        n_mids = self.get_n_mids(slider)
        mid_positions = slider.curve(np.linspace(0, 1, n_mids), return_array=True).astype(int)

        if with_background:
            img_shape = self.base_resolution_xy.astype(int)[::-1]
        else:
            min_x, min_y, max_x, max_y = self._get_slider_bbox_xyxy(mid_positions)
            img_shape = np.array((max_y - min_y, max_x - min_x)).astype(int)

            # 1. slider fade-in image
            # 2. slider image

            offset_xy = np.array([min_x, min_y])
            mid_positions = (mid_positions - offset_xy).astype(int)
        mid_img = np.zeros(img_shape, dtype=np.uint8)
        cv2.polylines(mid_img, [mid_positions], False, 255, 1, lineType=cv2.LINE_AA)

        # 3. slider fade-out image
        return mid_img

    def get_n_mids(self, slider: slider.Slider):
        """
        Returns the number of midpoints for a given slider type.

        Parameters:
        - slider: The slider object for which to calculate the number of midpoints.

        Returns:
        - n_mids: The number of midpoints for the given slider.

        """
        if isinstance(slider.curve, (Catmull, MultiBezier)):
            n_mids = self.n_mids_bezier
        elif isinstance(slider.curve, Perfect):
            n_mids = self.n_mids_curve
        elif isinstance(slider.curve, Linear):
            n_mids = slider.repeat + 1
        else:
            raise ValueError(f"unknown curve type: {type(slider.curve)}")
        return n_mids

    def mix_shapes_centering(self, arrays: "list[np.ndarray]", times: "list[int]"):
        """Mix multiple shapes into one array according to the times. Centering is applied."""
        assert len(arrays) == len(times), f"{len(arrays)} != {len(times)}"
        assert all([len(shape.shape) == 3 for shape in arrays]), "all shapes must be 3D"
        max_time = max([shape.shape[0] + t for shape, t in zip(arrays, times)])
        max_width = max([shape.shape[1] for shape in arrays])
        max_height = max([shape.shape[2] for shape in arrays])
        base_array = np.zeros((max_time, max_width, max_height), dtype=np.uint8)

        for x, t in zip(arrays, times):
            pad_w = (max_width - x.shape[1]) // 2
            pad_h = (max_height - x.shape[2]) // 2
            base_array[t:t + x.shape[0], pad_w:pad_w + x.shape[1], pad_h:pad_h + x.shape[2]] += x

        return base_array

    def mix_shapes(self, arrays, times, positions):
        """Mix multiple shapes into one array according to the times and positions.

        Args:
            arrays (list[np.ndarray]): The list of arrays representing shapes.
            times (list[int]): The list of start times for each shape.
            positions (list[tuple[int, int]]): The list of positions for each shape.

        Returns:
            np.ndarray: The mixed array of shapes.

        """
        shapes = np.stack([shape.shape for shape in arrays])
        max_T = (shapes[:, 0] + times).max()
        left_pad = positions[:, 0].max()
        right_pad = (shapes[:, 1] - positions[:, 0]).max()
        bottom_pad = positions[:, 1].max()
        top_pad = (shapes[:, 2] - positions[:, 1]).max()
        new_array = np.zeros((max_T, left_pad + right_pad, top_pad + bottom_pad), dtype=np.uint8)
        new_center = np.array([left_pad, bottom_pad])

        for obj, start_t, pos in zip(arrays, times, positions):
            offset = new_center - pos
            new_array[start_t:start_t + obj.shape[0], offset[0]:offset[0] + obj.shape[1],
                      offset[1]:offset[1] + obj.shape[2]] += obj

        return new_array

    def objects_to_array(self, hit_objects, normalize=False, cache_dir=None) -> np.ndarray:
        """
        Render hit objects to a 4D array.

        Args:
            hit_objects (List[Union[slider.Circle, slider.Slider, slider.Spinner]]): The list of hit objects to render.
            normalize (bool, optional): Whether to normalize the resulting array. Defaults to False.
            cache_dir (str, optional): The directory to cache the rendered arrays. Defaults to None.

        Returns:
            np.ndarray: The rendered hit objects as a 4D array.

        """
        _time_to_idx = partial(self.timedelta_to_ms, frame_ms=1000 / self.fps)

        last_object_endtime = np.max(
            [obj.time if isinstance(obj, slider.Circle) else obj.end_time for obj in hit_objects])
        n_frames = _time_to_idx(last_object_endtime) + 1
        shape = (len(self.channels_desc_dict), n_frames + self.circle_fade_out_frame,
                 *(self.base_resolution_xy // self.wh_scale + self.padding_xy * 2).astype(int))

        result_array = np.zeros(shape, dtype=np.uint8)
        _fill_circle_with_approach = partial(self._insert_object, result_array,
                                             self.circle_with_approach,
                                             offset_t=-self.approach_circle_frame)
        _fill_circle = partial(self._insert_object, result_array, self.circle,
                               offset_t=-self.approach_circle_frame)
        _fill_slider = partial(self._insert_slider,
                               result_array[self.channels_desc_dict["slider_body"]])
        _fill_spinner = partial(self._insert_spinner, result_array,
                                offset_t=-self.approach_circle_frame)
        for i_obj, obj in enumerate(hit_objects):
            if isinstance(obj, slider.Circle):
                idx = _time_to_idx(obj.time)
                _fill_circle_with_approach((self.channels_desc_dict["circle"], idx,
                                            *self._scale_and_pad(obj.position.x, obj.position.y)))

            elif isinstance(obj, slider.Slider):
                edge_times = obj.get_edgetimes()
                start, *repeats, end = edge_times
                idx_head = _time_to_idx(start)
                idx_tail = _time_to_idx(end)
                mid_positions = obj.calc_positions_at(np.linspace(0, 1, idx_tail - idx_head + 1))

                _fill_circle_with_approach((self.channels_desc_dict["slider_head"], idx_head,
                                            *self._scale_and_pad(obj.position.x, obj.position.y)))
                _fill_slider(obj, idx_head)
                _fill_circle((self.channels_desc_dict["slider_tail"], idx_tail,
                              *self._scale_and_pad(mid_positions[-1, 0], mid_positions[-1, 1])))

                # fill repeat points with circles
                for t_repeat in repeats:
                    idx_repeat = _time_to_idx(t_repeat)
                    _fill_circle((self.channels_desc_dict["slider_body"], idx_repeat,
                                  *self._scale_and_pad(mid_positions[idx_repeat - idx_head, 0],
                                                       mid_positions[idx_repeat - idx_head, 1])))

            elif isinstance(obj, slider.Spinner):
                idx_head = _time_to_idx(obj.time)
                idx_tail = _time_to_idx(obj.end_time)
                x, y = self._scale_and_pad(obj.position.x, obj.position.y)
                _fill_spinner(idx_tail - idx_head,
                              (self.channels_desc_dict["spinner_body"], idx_head, x, y))

            else:
                raise ValueError(f"unknown hitobject type: {type(obj)}")

        if normalize:
            result_array /= result_array.max()
        else:
            result_array[result_array > 255] = 255

        result_array = np.moveaxis(result_array, [0, 1, 2, 3], [0, 1, 3, 2])

        # cut to max_frame if longer, and repeat if shorter
        if self.max_frame:
            while result_array.shape[1] != self.max_frame:
                if result_array.shape[1] > self.max_frame:
                    result_array = result_array[:, :self.max_frame]
                else:
                    # note: this is bottleneck
                    result_array = np.tile(
                        result_array, (1, int(self.max_frame // result_array.shape[1]) + 1, 1, 1))

            assert result_array.shape[
                1] == self.max_frame, f"{result_array.shape} != {self.max_frame}"

        return result_array

    def save_video(self, vid_array: np.ndarray, path: "str | Path", map_array=True, audio_path=None,
                   fourcc="avc1", verbose=False):
        """
        Render objects as mp4 (x264) video.

        Args:
            vid_array (np.ndarray): The video array to be rendered.
            path (str | Path): The path to save the rendered video.
            map_array (bool, optional): Whether to map the video array before rendering. Defaults to True.
            audio_path (str | Path, optional): The path to the audio file to be added to the video. Defaults to None.
            fourcc (str, optional): The fourcc code for the video codec. Defaults to "avc1".
            verbose (bool, optional): Whether to display verbose output. Defaults to False.
        """
        if map_array:
            ch_groups = []
            for tgt in ["circle", "slider", "spinner"]:
                ch_groups.append([v for k, v in self.channels_desc_dict.items() if tgt in k])
            vid_array = np.stack([vid_array[tgt].max(0) for tgt in ch_groups], -1)

        T, H, W, C = vid_array.shape
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        out = cv2.VideoWriter(str(path), fourcc, self.fps, (W, H), isColor=C > 1)

        for i, frame in enumerate(vid_array):
            out.write(frame.astype(np.uint8))
        out.release()

        if audio_path is not None:
            add_audio_to_video(path, audio_path, verbose)

    def render(self, beatmap_path: "str | Path", output_path: "str | Path", fourcc="avc1"):
        """
        Render the given beatmap into a video file.

        Parameters:
        - beatmap_path (str | Path): The path to the beatmap file.
        - output_path (str | Path): The path to save the rendered video file.
        - audio_path (str | Path, optional): The path to the audio file. Defaults to None.

        """
        self(beatmap_path, output_path, fourcc)

    def __call__(self, beatmap_path: "str | Path", output_path: "str | Path", fourcc="avc1"):
        """
        Render the given beatmap into a video file.

        Parameters:
        - beatmap_path (str | Path): The path to the beatmap file.
        - output_path (str | Path): The path to save the rendered video file.
        - audio_path (str | Path, optional): The path to the audio file. Defaults to None.

        """

        beatmap = slider.Beatmap.from_path(beatmap_path)
        if beatmap.mode != slider.beatmap.GameMode.standard:
            raise ValueError(f"only standard mode is supported, but {beatmap.mode} is given.")
        beatmap_array = self.objects_to_array(beatmap.hit_objects())
        self.save_video(beatmap_array, output_path,
                        audio_path=Path(beatmap_path).parent / beatmap.audio_filename,
                        fourcc=fourcc)


def add_audio_to_video(video_path, audio_path, verbose=False):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            str(video_path),
        ],
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.DEVNULL if not verbose else None,
    )


def resize_video(path: str, out_path: str, w_final=None, h_final=None, pad_w=0, pad_h=0):
    """Resize the video by scaling and/or padding, using ffmpeg.

    Args:
        path (str): input video path
        out_path (str): output video path
        w (int, optional): final output video width. Defaults to None i.e. no size change.
        h (int, optional): final output video height. Defaults to None i.e. no size change.
        pad_w (int, optional): padding width included in the output video. Defaults to 0.
        pad_h (int, optional): padding height included in the output video. Defaults to 0.
    """
    assert not (w_final and h_final), "only one of w or h can be specified"

    if w_final:
        w_before_pad = w_final - pad_w * 2
        h_before_pad = -1
        h_final = f"{w_final}/a"
    elif h_final:
        h_before_pad = h_final - pad_h * 2
        w_before_pad = -1
        w_final = f"{h_final}*a"

    result = subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        path,
        f"-vf",
        f"scale={w_before_pad}:{h_before_pad}:force_original_aspect_ratio=1,pad={w_final}:{h_final}:{pad_w}:{pad_h}",
        "-c",
        "copy",
        "-crf",
        "23",
        "-preset",
        "slow",
        out_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)
