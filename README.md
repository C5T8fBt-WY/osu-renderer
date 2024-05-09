# osu-renderer

## Overview

This library is a renderer (visualizer) of osu files used in the rhythm game osu, currently for its std rule.

Its concept is to render a beatmap in literally one second. To do this, objects are represented by simple shapes and the current resolution is not really high.

## Installation

1. Prerequisite: FFmpeg with and Python>=3.7 (tested in Python 3.8) 
2. git clone https://github.com/C5T8fBt-WY/osu-renderer
3. cd osu-renderer
4. pip install -e .


## Usage

``` Python
from osu-renderer import StandardBeatmapRenderer

beatmap_path = "./my_awesome_beatmap.osu"
output_path = "result.mp4"
renderer = StandardBeatmapRenderer()
renderer(beatmap_path, output_path)
```

## Notice
- Catmul slider, which was often used in old beatmaps, is currently not well rendered. It will be fixed soon.