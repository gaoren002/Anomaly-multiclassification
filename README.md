# Anomaly Multi-classification in Industrial Scenarios: Transferring Few-shot Learning to a New Task
This repository is based on PatchCore's official implementation.
This repository contains the implementation proposed in our paper.

This repository also doesn't provide a evaluation process, all results can be trained and tested very soon so we only save the model being pretrained.

---
## Requirements

Our results were computed using Python 3.8, with packages and respective version noted in
`requirements.txt`. In general, the majority of experiments should not exceed 24GB of GPU memory;


## Quick Guide
We recommand you to use Visual Studio Code, as we provide a "launch.json" file to launch our project.

All settings can be found in "launch.json" and run_patchcore.py

### Setting up DTD
Download the DTD from here:<https://www.robots.ox.ac.uk/~vgg/data/dtd/>
Make sure that it follows the following data tree:
```shell
generate_anomaly_pkg
|-- dtd
|-----|----- images
|-----|----- imdb
|-----|----- labels
|-- generate_anomaly.py
|-- data_loader_for_draem.py
|-- ...
```
### Setting up MVTec AD and generate synthesized anomaly 

To set up the main MVTec AD benchmark, download it from here: <https://www.mvtec.com/company/research/datasets/mvtec-ad>.
 Make sure that it follows the following data tree:

```shell
mvtec_anomaly_detection
|-- bottle
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ broken_large
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
|-- cable
|-- generate_foreground.py
|-- generate_foreground copy.py
|-- ...
```

containing in total 15 subdatasets: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

Then you can run "generate_foreground.py" and "generate_foreground copy.py" respectively with "run Currrent file" in "launch.json", the generated image will be placed in folder "generate_anomaly_pkg"

### Training
run ./bin/run_patchcore.py with "run_patchcore"

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the MIT License.
