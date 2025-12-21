import torch
from data.video import save_video
from wan_loader import load_wan_pipe
from models.set_condition_branch import set_stand_in
from preprocessor import FaceProcessor
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ip_image",
    type=str,
    default="test/input/lecun.jpg",
    help="Input face image path or URL",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="一位男性舒适地坐在书桌前，正对着镜头，仿佛在与屏幕前的亲友对话。他的眼神专注而温柔，嘴角带着自然的笑意。背景是他精心布置的个人空间，墙上贴着照片和一张世界地图，传达出一种亲密而现代的沟通感。",
    help="Text prompt for video generation",
)
parser.add_argument(
    "--output", type=str, default="test/output/lecun.mp4", help="Output video file path"
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for reproducibility"
)
parser.add_argument(
    "--num_inference_steps", type=int, default=20, help="Number of inference steps"
)
parser.add_argument(
    "--wan_version", type=str, default="2.1", help="Wan version"
)

parser.add_argument(
    "--negative_prompt",
    type=str,
    default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    help="Negative prompt to avoid unwanted features",
)
parser.add_argument("--tiled", action="store_true", help="Enable tiled mode")
parser.add_argument(
    "--fps", type=int, default=25, help="Frames per second for output video"
)
parser.add_argument(
    "--quality", type=int, default=9, help="Output video quality (1-9)"
)
parser.add_argument(
    "--base_path",
    type=str,
    default="checkpoints/Wan2.1/t2v",
    help="Path to base model checkpoint",
)
parser.add_argument(
    "--stand_in_path",
    type=str,
    default="checkpoints/Stand-In/",
    help="Path to LoRA weights checkpoint",
)
parser.add_argument(
    "--antelopv2_path",
    type=str,
    default="checkpoints/antelopev2",
    help="Path to AntelopeV2 model checkpoint",
)
args = parser.parse_args()


face_processor = FaceProcessor(antelopv2_path=args.antelopv2_path)
ip_image = face_processor.process(args.ip_image)

pipe = load_wan_pipe(base_path=args.base_path, wan_version=args.wan_version, torch_dtype=torch.bfloat16)

set_stand_in(
    pipe,
    model_path=args.stand_in_path,
    wan_version=args.wan_version
)

video = pipe(
    prompt=args.prompt,
    negative_prompt=args.negative_prompt,
    seed=args.seed,
    ip_image=ip_image,
    num_inference_steps=args.num_inference_steps,
    tiled=args.tiled,
)
save_video(video, args.output, fps=args.fps, quality=args.quality)
