from src.transformers import AutoProcessor
from huggingface_hub import list_models, hf_hub_download, HfApi
import json
import warnings

warnings.filterwarnings("ignore")

models = list_models(tags="object-detection", sort="downloads", limit=1000)

count = 0
for model in models:
    try:
        config_path = hf_hub_download(repo_id=model.id, filename="preprocessor_config.json")
        with open(config_path, "r") as file:
            config = json.load(file)

        if not isinstance(config.get("size"), dict):
            image_processor = AutoProcessor.from_pretrained(model.id)
            commit_description = ""

            new_config = {}
            for key, value in config.items():
                if "feature_extractor_type" == key:
                    new_config["image_processor_type"] = image_processor.__class__.__name__
                    commit_description += f"- Replace deprecated `\"feature_extractor_type\": \"{value}\"` with `\"image_processor_type\": \"{image_processor.__class__.__name__}\"`\n"
                elif "max_size" == key:
                    if "size" not in new_config:
                        new_config["size"] = {}
                        commit_description += "- Remove deprecated `max_size` and replace `size` with a `Dict`\n"
                    new_config["size"]["longest_edge"] = value
                elif "size" == key:
                    if "size" not in new_config:
                        new_config["size"] = {}
                        commit_description += "- Remove deprecated `max_size` and replace `size` with a `Dict`\n"
                    new_config["size"]["shortest_edge"] = value
                else:
                    new_config[key] = value

            with open("preprocessor_config.json", 'w') as file:
                json.dump(new_config, file, indent=2)

            api = HfApi()
            results = api.upload_file(
                path_or_fileobj="preprocessor_config.json",
                path_in_repo="preprocessor_config.json",
                repo_id=model.id,
                repo_type="model",
                commit_message="[Clean-up] Planned removal of the `max_size` argument",
                commit_description=commit_description,
                create_pr=True
            )

            print(f"#{count} Model: {model.id}")
            print(f"Created: {model.created_at}")
            print(f"Downloads: `{model.downloads}`")
            print(f"PR: [[Clean-up] Planned removal of the `max_size` argument]({results.commit_url})")
            count += 1

        if count >= 5:
            break

    except Exception as e:
        pass







