import tensorflow as tf
import argparse
from contextlib import nullcontext
import yaml
from animate import animate
from utils import load_image_video_pair, save_video,load_models_tflite

parser = argparse.ArgumentParser(description="Run inference")
parser.add_argument('--target', choices=['direct', 'savedmodel', 'tflite'], default='tflite',
                    help="model version to run (between running the model directly, running the model's saved_model, and running its converted tflite")
parser.add_argument('--mode', choices=['animate', 'reconstruction',], default='animate', help="Run mode (animate, reconstruct, or train)")
parser.add_argument('--datamode', choices=['file', 'dataset'], default='file', help='Data input mode (CLI-given file or config-defined dataset)')
parser.add_argument("--model", action="store", type=str, default="vox-adv", help="model name")
parser.add_argument("--source_image", action="store", type=str, default="example/img.png", help="source image path for file datamode")
parser.add_argument("--driving_video", action="store", type=str, default="example/crop2.mp4", help="driving video path for file datamode")
parser.add_argument("--output", action="store", type=str, default="example/output", help="output file name")
parser.add_argument("--dontappend",  action="store_true", help="don't append format name and .mp4 to the output filename")
parser.add_argument("--relative", action="store_true", help="relative kp mode")
parser.add_argument("--adapt", dest="adapt_movement_scale", action="store_true", help="adapt movement to the proportion between the sizes of subjects in the input image and the driving video")
parser.add_argument("--frames", type=int, default=-1, help="number of frames to process")
parser.add_argument("--batchsize", dest="batch_size", type=int, default=4, help="batch size")
parser.add_argument("--exactbatch", dest="exact_batch", action="store_true", help="force static batch size, tile source image to batch size")
parser.add_argument("--device", dest="device", default=None, help="device to use")
parser.add_argument("--profile", action="store_true", help="enable tensorboard profiling")
parser.add_argument("--visualizer", action="store_true", help="enable visualizer, only relevant for dataset datamode")
parser.add_argument('--loadwithtorch', action="store_true",
                    help="use torch to load checkpoints instead of trying to load tensor buffers manually (requires pytorch)")

parser.set_defaults(relative=True)
parser.set_defaults(adapt=True)
parser = parser.parse_args()

if parser.loadwithtorch:
    import load_torch_checkpoint
    load_torch_checkpoint.mode = 'torch'

context = tf.device(parser.device) if parser.device is not None else nullcontext()
if parser.profile:
    tf.debugging.set_log_device_placement(True)

load_funcs = {'tflite':load_models_tflite}

config_path = f"config/{parser.model}-256.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)
frame_shape = config['dataset_params']['frame_shape']
num_channels = config['model_params']['common_params']['num_channels']

with context:
    kp_detector, process_kp_driving, generator, _interpreter_obj_list = load_funcs[parser.target](parser.model, prediction_only=parser.datamode=='file', static_batch_size = None if not parser.exact_batch else parser.batch_size, hardcode='1' + str(int(parser.adapt_movement_scale)))
    format_appends = {'tflite':'.tflite'}

    if parser.mode == 'animate':
        if parser.datamode == 'file':
            source_image, frames, fps = load_image_video_pair(parser.source_image, parser.driving_video, frames=parser.frames, frame_shape=frame_shape, num_channels=num_channels)
            predictions, _ = animate(source_image, frames, generator, kp_detector, process_kp_driving, 
                                parser.relative, parser.relative, parser.adapt_movement_scale,
                                batch_size=parser.batch_size, exact_batch=parser.exact_batch, profile=parser.profile)
            output = parser.output
            if not parser.dontappend:
                output = output + format_appends[parser.target] + '.mp4'
            save_video(output, predictions, fps=fps)

print("Done.")
