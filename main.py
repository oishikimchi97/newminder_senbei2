import argparse
import glob
from pathlib import Path
from types import SimpleNamespace

from substraction import substract
from infer import infer


def main(args):
    input_img_paths = glob.glob(args.input_dir + "/*")
    sub_img_paths = []
    Path(args.output_dir).mkdir(exist_ok=True)

    for input_img_path in input_img_paths:
        sub_img_path = Path(args.output_dir) / Path(input_img_path).name
        sub_img_path = str(sub_img_path)
        sub_img_paths.append(sub_img_path)

        substract_args = SimpleNamespace(
            template_dir=args.template_dir,
            base_img_path=args.base_img_path,
            target_img_path=input_img_path,
            sub_path=sub_img_path
        )
        substract(substract_args)
    for sub_img_path in sub_img_paths:
        infer_args = SimpleNamespace(
            target_img_path=sub_img_path,
            output_dir=args.output_dir
        )
        infer(infer_args)
    print("process was completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--base_img_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--template_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)
