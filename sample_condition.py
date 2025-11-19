import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import transforms

from data.dataloader import get_dataloader, get_dataset
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from util.tools import rgb_to_gray


def load_yaml(file_path: Path) -> dict:
    with file_path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--diffusion_config", type=str)
    parser.add_argument("--task_config", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./results")

    args = parser.parse_args()

    model_config_file_path = Path(args.model_config)
    diffusion_config_file_path = Path(args.diffusion_config)
    task_config_file_path = Path(args.task_config)
    save_directory_path = Path(args.save_dir)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device_log = f"Device set to {device_str}."
    logger.info(device_log)
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(model_config_file_path)
    diffusion_config = load_yaml(diffusion_config_file_path)
    task_config = load_yaml(task_config_file_path)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuration."

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    operation_message = (
        f"Operation: {measure_config['operator']['name']} / "
        "Noise: {measure_config['noise']['name']}"
    )
    logger.info(operation_message)

    # Prepare conditioning method
    cond_config = task_config["conditioning"]
    cond_method = get_conditioning_method(
        cond_config["method"], operator, noiser, **cond_config["params"]
    )
    measurement_cond_fn = cond_method.conditioning
    conditioning_method_message = (
        f"Conditioning method : {task_config['conditioning']['method']}"
    )
    logger.info(conditioning_method_message)

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(
        sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn
    )

    # Working directory
    out_path = save_directory_path / measure_config["operator"]["name"]
    for img_dir in ["input", "recon", "progress", "label"]:
        (out_path / img_dir).mkdir(parents=True, exist_ok=True)

    # Prepare dataloader
    data_config = task_config["data"]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    mask_gen = None
    if measure_config["operator"]["name"] == "inpainting":
        mask_gen = mask_generator(**measure_config["mask_opt"])

    # Do Inference
    for i, (ref_img, raw_min, raw_max) in enumerate(loader):
        inference_message = f"Inference for image {i}"
        logger.info(inference_message)
        output_file_name = str(i).zfill(3) + ".raw"
        ref_img_device = ref_img.to(device)

        # Exception) In case of inpainting,
        if measure_config["operator"]["name"] == "inpainting" and mask_gen is not None:
            mask = mask_gen(ref_img_device)
            if mask is None:
                runtime_error_msg = (
                    "mask_generator returned None;"
                    " check measurement.mask_opt and mask_generator implementation."
                )
                raise RuntimeError(runtime_error_msg)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img_device, mask=mask)
            y_n = noiser(y)

        else:
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img_device)
            y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img_device.shape, device=device).requires_grad_()
        sample = sample_fn(
            x_start=x_start, measurement=y_n, record=True, save_root=str(out_path)
        )

        input_np = rgb_to_gray(clear_color(y_n))
        label_np = rgb_to_gray(clear_color(ref_img_device))
        recon_np = rgb_to_gray(clear_color(sample))

        recon_np = (recon_np - recon_np.min()) / (
            recon_np.max() - recon_np.min() + 1e-8
        )
        recon_np = recon_np * (raw_max.item() - raw_min.item()) + raw_min.item()

        input_np.astype(np.float64).tofile(out_path / "input" / output_file_name)
        label_np.astype(np.float64).tofile(out_path / "label" / output_file_name)
        recon_np.astype(np.float64).tofile(out_path / "recon" / output_file_name)


if __name__ == "__main__":
    main()
