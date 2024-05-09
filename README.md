# osu-renderer

## Overview

This library is a renderer (visualizer) of osu files used in the rhythm game osu, currently for its std rule.



https://github.com/C5T8fBt-WY/osu-renderer/assets/72484917/0f3d886f-1248-4ca3-8fa8-0a23022c9621



Its concept is to render a beatmap in literally one second. To do this, objects are represented by simple shapes and the current resolution is not really high.

## Installation

Prerequisite: FFmpeg with and Python>=3.7 (tested in Python 3.8) 
```
git clone https://github.com/C5T8fBt-WY/osu-renderer
cd osu-renderer
pip install .  # or add `-e` for editable mode
```


## Usage

``` Python
from osu-renderer import StandardBeatmapRenderer

beatmap_path = "./my_awesome_beatmap.osu"
output_path = "result.mp4"
renderer = StandardBeatmapRenderer()
renderer.render(beatmap_path, output_path)
```

## Notice
- Catmul slider, which was often used in old beatmaps, is currently not well rendered. It will be fixed soon.
