# Newminder Senbei Inspection Project in RUTILEA


## Data Setup

```shell
$ python data.py --senbei_template senbei_data/template.bmp --senbei_target senbei_data/data4.bmp --x_position 500 --width_start 1 --width_end 100 --step_num 5 --output_dir output_dir
```

### Params
**senbei_template**: the template image file path whose image is neatly cropped for template matching. \
**senbei_target**: the target image file path for cropping and making data \
**x_position**: the x coordinate which the program draw lines to \
**width_start**: the initial width about drawing lines \
**width_end**: the end of width about drawing lines \
**step_num**: the step size between the widths \
**output_dir**: the output folder path

<hr>

## Subtraction
```shell
$ python subtraction.py --template_dir senbei_data/infer_template --base_img_path senbei_data/template.bmp --target_img_path output_dir/drawed_senbei_line_41.bmp --sub_path sub.bmp
```

### Params
**template_dir**: a folder path which contains the template images such as a nose, and eyes of a senbei \
**base_img_path**: the base senbei image file path \
**target_img_path**: the target senbei image file path \
**sub_path**: the output file path 

## Inference

<hr>

```shell
$ python infer.py --target_img_path sub.bmp --output_path inferred.bmp
```

### Params
**target_img_path**: the target image file path subtracted by subtraction.py \
**output_dir**: the output folder path. The output file will be saved into this folder with the same target_img_path file name

## Main

<hr>

```shell
$ python main.py --input_dir output_dir --base_img_path senbei_data/template.bmp --template_dir senbei_data/infer_template --output_dir output
```

### Parmas

**input_dir**: the input directory path which contains the inspection target images file. \
The target images corresponds to the files created by data script \
**base_img_path**: the base image file path \
**template_dir**: the directory path of template files \
**output_dir**: the output directory path
