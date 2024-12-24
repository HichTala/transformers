import sys
from contextlib import contextmanager

from src.transformers import AutoProcessor
from huggingface_hub import list_models, hf_hub_download, HfApi
import json
import warnings

warnings.filterwarnings("ignore")


@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open('/dev/null', 'w')  # Redirect stdout
        sys.stderr = open('/dev/null', 'w')  # Redirect stderr
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


models = list_models(tags="object-detection", sort="downloads", limit=1000)

already_processed = ['microsoft/table-transformer-detection', 'microsoft/table-transformer-structure-recognition',
                     'TahaDouaji/detr-doc-table-detection', 'Aryn/deformable-detr-DocLayNet',
                     'valentinafeve/yolos-fashionpedia', 'isalia99/detr-resnet-50-sku110k',
                     'aParadigmP/table-transformer-detection-custom-ale',
                     'nickmuchi/yolos-small-finetuned-license-plate-detection', 'facebook/detr-resnet-101-dc5',
                     'facebook/detr-resnet-50-dc5', 'nickmuchi/yolos-small-rego-plates-detection',
                     'hilmantm/detr-traffic-accident-detection', 'Omnifact/conditional-detr-resnet-101-dc5',
                     'nickmuchi/yolos-small-plant-disease-detection', 'nickmuchi/detr-resnet50-license-plate-detection']

number_of_prs = 30
count = 0
for model in models:
    try:
        if model.downloads < 100:
            break

        if model.id not in already_processed:
            with suppress_output():
                config_path = hf_hub_download(repo_id=model.id, filename="preprocessor_config.json")
                with open(config_path, "r") as file:
                    config = json.load(file)

            if not isinstance(config.get("size"), dict):
                with suppress_output():
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
                already_processed.append(model.id)

                print(f"- [ ] {count} Model: {model.id}")
                print(f"Created: {model.created_at}")
                print(f"Downloads: `{model.downloads}`")
                print(f"PR: [[Clean-up] Planned removal of the `max_size` argument]({results.pr_url})")
                print()
                count += 1

            if count >= number_of_prs:
                break

    except Exception as e:
        pass

print(already_processed)
