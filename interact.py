import scripts.txt2img as sampler

if __name__ == "__main__":

    print("Loading models, hold on...")
    model = sampler.load_model_from_config("configs/stable-diffusion/v1-inference.yaml", "models/ldm/stable-diffusion-v1/model.ckpt")

    print("Welcome to stable diffusion interact prompt mode! ENter your prompt to generate a sample!")
    while True:
        prompt = input(">> ")

        sampler.sample(
            prompt=prompt,
            model=model
        )
