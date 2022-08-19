import scripts.txt2img as sampler
from omegaconf import OmegaConf

if __name__ == "__main__":

    print("Starting model loading...")
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = sampler.load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

    print("Welcome to stable diffusion interact prompt mode! Enter your prompt to generate a sample!")
    while True:
        prompt = input(">> ")

        sampler.sample(
            prompt=prompt,
            model=model
        )
