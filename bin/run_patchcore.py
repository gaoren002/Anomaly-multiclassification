import contextlib
import logging
import os
import sys
import glob

import click
import numpy as np
import torch
sys.path.append("./src")
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from itertools import combinations
import wandb
from torch.utils.tensorboard import SummaryWriter
import random
LOGGER = logging.getLogger(__name__)

NUM_ANOMALY_CLASSES = 10 # If defined，works for pretrain_dataset, otherwise the number of anomaly of pretrain_dataset is the same as the corresonding real anomaly dataset 
COMBINATION_TIMES = 1000 # Before pretaining,  COMBINATION_TIMES pretrain datasets will be generated, all of which contains different pseudo defects, it will take some time. So before pretraining set an adequate  COMBINATION_TIMES
THRESHOLD_STOP = 0.4 # a accuracy threshold to stop pretraining in order to avoid overfitting. 
LOAD_MODEL = False  # if set True, skip pretrainng or loading the pretrained model. 
SHOT_NUM = 2 # only set it as 1,2,3,4,5 
Running_times = 45 # running times refered in paper(actually epoch)
LOAD_PATCHCORE_PATH = None # You can load the patchcore model so as to avoid subsampling
DIRECT_TRAIN = False # if set True, LOAD_MODEL will be ignored
train_contrastive_classification = True  # if set, contrastive classifier will be trained and tested.
relation_net = False # if set, vanilla baseline will be trained and tested.Do not set train_contrastive_classification and relation_net simultaneously, it will raise bugs. 
if_pretrained_net_for_backbone = True # if set False, you can refer to a specified ResNet for PatchCore, but they always performs worse. 
if_direct_classfication = True # if set, it will try to fine-tune the fc layer of a ResNet(used for classification not for patchcore), whose input is the raw image rather than features
MAML = False # if set, using maml strategy when pretrain contrastive classifier, only useful when train contrastive classifier.
NAME = "transistor" # set it so as you can load a specified model, and the name of pretrained model to be saved is relevent to this name, if skip pretraining, just ignore this.
_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],"mvtec_3d":["patchcore.datasets.mvtec_3d","MVTec3DDataset"]}
_DATASETS_PRETRAIN = {"generated_mvtec": ["patchcore.datasets.generated_mvtec", "GeneratedMVTecDataset", "GeneratedMVTecDatasetForRelationNet"]
,"generated_mvtec_3d": ["patchcore.datasets.generated_mvtec_3d", "GeneratedMVTec3DDataset", "GeneratedMVTec3DDatasetForRelationNet"]}

@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    wandb.init(project="defect-classfication",name=NAME+"_"+str(THRESHOLD_STOP)+"_shot"+str(SHOT_NUM))
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):  
    # heartrate.trace(browser=True)
    if_mlp_classification = False # mlp deprecated, so setting it False
    torch.use_deterministic_algorithms(True)
    device = patchcore.utils.set_torch_device(gpu)
    patchcore.utils.fix_seeds(seed)
    writer = SummaryWriter(log_dir="./logs/"+NAME+"_"+str(THRESHOLD_STOP))
    # torch.use_deterministic_algorithms(True)
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )


    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    result_collect = []
    list_of_dataloaders = methods["get_dataloaders"](seed)
    
    class_anomaly_subclass_number = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders): # list_of_dataloaders:For example, all item categories in MVTec.
        dataset = dataloaders["testing"].dataset
        anomaly_classes = []
        for i in range(len(dataset)):
            anomaly_classes.append(dataset.data_to_iterate[i][1])
        anomaly_classes_number = len(set(anomaly_classes))   #TODO 
        class_anomaly_subclass_number.append(anomaly_classes_number-1)  # Count the number of defect types for each item category, excluding the "good" category.
    if  DIRECT_TRAIN or LOAD_MODEL:
        list_of_dataloaders_pretrain, list_of_dataloaders_pretrain_relation_net = [{}for i in range(len(list_of_dataloaders))],[{}for i in range(len(list_of_dataloaders))]
       
    else:
        list_of_dataloaders_pretrain, list_of_dataloaders_pretrain_relation_net = methods["get_dataloaders_pretrain"](seed, combination_times=COMBINATION_TIMES,class_anomaly_subclass_number=class_anomaly_subclass_number)
          # for dataloader_count, dataloaders_pretrain in enumerate(list_of_dataloaders_pretrain):
    #     dataset = dataloaders_pretrain["training"].dataset
    #     anomaly_classes = []
    #     for i in range(len(dataset)):
    #         anomaly_classes.append(dataset.data_to_iterate[i][1])
    #     anomaly_classes_number = len(set(anomaly_classes))
    for dataloader_count, item in enumerate(zip(list_of_dataloaders,list_of_dataloaders_pretrain,list_of_dataloaders_pretrain_relation_net)):
        # list_of_dataloaders different categoried of items
        # every list contains 3 dataloaders，respectively training, validation, testing，
        # The returned data will be a dictionary, containing the types of anomalies.
        dataloaders, dataloaders_pretrain, dataloaders_pretrain_relation_net = item  # the last two  is the list of dataloaders
        
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        # patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name  # mvtec 或者  mvtec_物品种类

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device,
                                                      class_anomaly_subclass_number[dataloader_count],
                                                      writer = writer,
                                                      if_pretrained_net_for_backbone=if_pretrained_net_for_backbone, 
                                                      if_direct_classfication=if_direct_classfication, 
                                                      if_mlp_classification=False, # mlp, deprecated,,relevent code deprecated
                                                      if_pretrained_net_for_direct_classification=True, # whether to use the pretrained resnet to directly classify the defect image or customed resnet
                                                      train_metric_classification=False,  # metric learning, deprecated,relevent code deprecated
                                                      train_contrastive_classification=train_contrastive_classification,
                                                      relation_net=relation_net,
                                                      if_backbone_aggregation=False)  # deprecated, only used when I try to combine two results respectively from a pretrained ResNet and a ResNet using Spark(a training method like MAE)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if isinstance(PatchCore.backbone, list):
                    if PatchCore.backbone[0].seed is not None:
                        patchcore.utils.fix_seeds(PatchCore.backbone[0].seed, device)
                else:
                    if PatchCore.backbone.seed is not None:
                        patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                if LOAD_PATCHCORE_PATH:
                    PatchCore.load_from_path(load_path=LOAD_PATCHCORE_PATH,device="cuda",nn_method=patchcore.common.FaissNN(True, 0),
                                           if_pretrained_net_for_backbone=True)
                else:
                    
                    PatchCore.fit(dataloaders["training"])  # The dataloader for training is responsible for extracting Patchcore's memory bank.
                # PatchCore.train_foreground(dataloaders["training"])
                if not DIRECT_TRAIN:
                    if LOAD_MODEL:
                        saved_model = torch.load("./saved_models/"+NAME+"_"+str(THRESHOLD_STOP)+".pt")
                        if train_contrastive_classification:
                            PatchCore.contrastive_model = saved_model["model"]
                            PatchCore.optimizers_contrastive = saved_model["optimizer"]#torch.optim.Adam(PatchCore.contrastive_model.parameters(), lr=0.0003)
                        if relation_net:
                            PatchCore.relation_model = saved_model["model"]
                            PatchCore.optimizers_relation = saved_model['optimizer']
                        if if_mlp_classification:
                            PatchCore.classfication_mlp_model = saved_model["model"]
                            PatchCore.optimizer_mlp = saved_model["optimizer"]
                    else:
                        for i in range(COMBINATION_TIMES): # dataloaders_pretrain is a list containing combination_times number of dataloaders (corresponding to different categories of pseudo defects).
                            result = PatchCore.pre_train(dataloaders_pretrain[i]["training"],dataloaders_pretrain_relation_net[i]["training"], i-1, MAML)  # pretrain with pseudo categories of anomaly
                            if i % 30 ==29:
                                
                                if relation_net:
                                    PatchCore.predict_for_relation_net(dataloaders["testing"],shot_num=SHOT_NUM)
                                if train_contrastive_classification:
                                    PatchCore.predict_contrastive(dataloaders["testing"],shot_num=SHOT_NUM)
                                if if_mlp_classification:
                                    PatchCore.predict_for_classification_mlp(dataloaders["testing"], shot_num=shot_num)

                            if result>THRESHOLD_STOP:
                                if relation_net:
                                    PatchCore.predict_for_relation_net(dataloaders["testing"],last_pretrain=True,shot_num=SHOT_NUM)
                                if train_contrastive_classification:
                                    PatchCore.predict_contrastive(dataloaders["testing"],last_pretrain=True,shot_num=SHOT_NUM)#last_pretrain is used to record the last time of pre-training.
                                if if_mlp_classification:
                                    PatchCore.predict_for_classification_mlp(dataloaders["testing"],last_pretrain=True, shot_num=shot_num)  
                                break
                        if not os.path.exists("./saved_models"):
                            os.mkdir("./saved_models")
                        if relation_net:
                            torch.save({'model':PatchCore.relation_model,"optimizer":PatchCore.optimizers_relation},"./saved_models/"+NAME+"_"+str(THRESHOLD_STOP)+".pt")
                        if train_contrastive_classification:
                            torch.save({'model':PatchCore.contrastive_model,"optimizer":PatchCore.optimizers_contrastive},"./saved_models/"+NAME+"_"+str(THRESHOLD_STOP)+".pt")
                        # torch.save({'model':PatchCore.relation_model,"optimizer":PatchCore.optimizers_relation},"./saved_models/"+NAME+"_"+str(THRESHOLD_STOP)+".pt")
                        if if_mlp_classification:
                            torch.save({"model":PatchCore.classfication_mlp_model,"optimizer":PatchCore.optimizer_mlp},"./saved_models/"+NAME+".pt")
                    #torch.cuda.empty_cache()
                PatchCore.train(dataloaders["testing"],SHOT_NUM,Running_times)
                if relation_net:
                    PatchCore.predict_for_relation_net(dataloaders["testing"],last=True,shot_num=SHOT_NUM)
                if train_contrastive_classification:
                    PatchCore.predict_contrastive(dataloaders["testing"],last=True,shot_num=SHOT_NUM)#  last_pretrain is used to record the last time of pre-training. # last: final prediction
                if if_mlp_classification:
                    PatchCore.predict_for_classification_mlp(dataloaders["testing"],last=True, shot_num=shot_num)
            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"],SHOT_NUM
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=0)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device, class_anomaly_subclass_number,writer,
                      if_pretrained_net_for_backbone, if_direct_classfication, if_mlp_classification,
                      if_pretrained_net_for_direct_classification,
                      train_metric_classification, train_contrastive_classification,relation_net, if_backbone_aggregation):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll  # multiple patchcore，from multiple patchcore_backbone
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            if not if_backbone_aggregation:
                if if_pretrained_net_for_backbone:
                    backbone = patchcore.backbones.load(backbone_name, True)
                    # ckpt_for_backbone = torch.load('./output_180_3.pth')
                    # backbone.load_state_dict(ckpt_for_backbone, strict=False)
                else:
                    backbone = patchcore.backbones.load(backbone_name, False)
                    ckpt_for_backbone = torch.load('./resnet50_pretrain2representation2.pth')
                    backbone.load_state_dict(ckpt_for_backbone, strict=False)
                backbone.name, backbone.seed = backbone_name, backbone_seed
            else:
                LOGGER.info("Parameter if_pretrained_net_for_backbone will not work because if_backbone_aggregation is set")
                backbone1 = patchcore.backbones.load(backbone_name, True)
                backbone1.name, backbone1.seed = backbone_name, backbone_seed
                backbone2 = patchcore.backbones.load(backbone_name, False)
                backbone2.name, backbone2.seed = backbone_name, backbone_seed
                ckpt_for_backbone2 = torch.load('./resnet50_pretrain2representation2.pth')
                backbone2.load_state_dict(ckpt_for_backbone2, strict=False)
                backbone = [backbone1, backbone2]
                
            if if_direct_classfication:
                if if_pretrained_net_for_direct_classification:
                    backbone_for_direct_classification = patchcore.backbones.load(backbone_name, True)
                else:
                    ckpt_for_backbone_train = torch.load('./resnet50_pretrain2representation2.pth')
                    backbone_for_direct_classification = patchcore.backbones.load(backbone_name, False)
                    backbone_for_direct_classification.load_state_dict(ckpt_for_backbone_train, strict=False)
                backbone_for_direct_classification.name, backbone_for_direct_classification.seed = backbone_name, backbone_seed
            else:
                backbone_for_direct_classification = None

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                backbone_for_direct_classification=backbone_for_direct_classification,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                writer=writer,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                class_anomaly_subclass_number=class_anomaly_subclass_number,
                if_direct_classfication=if_direct_classfication,
                if_mlp_classification=if_mlp_classification,
                f=train_metric_classification,
                train_contrastive_classification=train_contrastive_classification,
                relation_net=relation_net
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=0, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:  # subdataset means items ,dateset means mvtec or ohthers
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )                

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                # "pre_train": pre_train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader
            }

            dataloaders.append(dataloader_dict)  # dataloaders contains all types of items
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


@main.command("pretrain_dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("ori_data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1.0, show_default=True)
@click.option("--batch_size", default=9, type=int, show_default=True)
@click.option("--num_workers", default=0, type=int, show_default=True)
@click.option("--resize", default=512, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def pretrain_dataset(
    name,
    data_path,
    ori_data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS_PRETRAIN[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1],dataset_info[2]])
    pass
    def get_dataloaders_pretrain(seed, combination_times, class_anomaly_subclass_number):
        dataloaders_list = [] 
        dataloaders_list_relation_net = []
        paths = glob.glob(data_path+"/*")
        anomaly_list = os.listdir(paths[0])
        for ii,subdataset in enumerate(subdatasets):
            dataloaders_anomaly_list = []
            dataloaders_anomaly_list_relation_net = []
            all_combinations = []

            for i in range(10000):
                if NUM_ANOMALY_CLASSES:
                    all_combinations.append(random.sample(anomaly_list, NUM_ANOMALY_CLASSES))
                else:
                    all_combinations.append(random.sample(anomaly_list, class_anomaly_subclass_number[ii]))
            for l in range(combination_times):
                if l >= combination_times:
                    break
                random_idx =  np.random.randint(0,len(all_combinations)-1)
                selected_anomaly_class = all_combinations[random_idx]
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    ori_data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=1.0,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    selected_anomaly_class=selected_anomaly_class,
                    seed=seed,
                    augment=augment,
                )                

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True
                )

                train_dataloader.name = name
                if subdataset is not None: #  If a specific item category is specified, then attach the category subdataset to the name attribute.
                    train_dataloader.name += "_" + subdataset
                    val_dataloader = None
                dataloader_dict = {
                    "training": train_dataloader,
                    "validation": val_dataloader
                }
                
                
                
                
                train_dataset_relation_net = dataset_library.__dict__[dataset_info[2]](
                    data_path,
                    ori_data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    selected_anomaly_class=selected_anomaly_class,
                    seed=seed,
                    augment=augment,
                )                

                train_dataloader_relation_net = torch.utils.data.DataLoader(
                    train_dataset_relation_net,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True
                )

                train_dataloader_relation_net.name = name
                if subdataset is not None: #  If a specific item category is specified, then attach the category subdataset to the name attribute.
                    train_dataloader_relation_net.name += "_" + subdataset

                if train_val_split < 1:
                    val_dataset_relation_net = dataset_library.__dict__[dataset_info[2]](
                        data_path,
                        ori_data_path,
                        classname=subdataset,
                        resize=resize,
                        train_val_split=train_val_split,
                        imagesize=imagesize,
                        split=dataset_library.DatasetSplit.VAL,
                        selected_anomaly_class=selected_anomaly_class,
                        seed=seed,
                    )

                    val_dataloader_relation_net = torch.utils.data.DataLoader(
                        val_dataset_relation_net,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                else:
                    val_dataloader_relation_net = None
                dataloader_dict_relation_net = {
                    "training": train_dataloader_relation_net,
                    "validation": val_dataloader_relation_net
                }
                dataloaders_anomaly_list.append(dataloader_dict)  # combination_times个dataloader
                dataloaders_anomaly_list_relation_net.append(dataloader_dict_relation_net)
            dataloaders_list.append(dataloaders_anomaly_list)  # wht number of dataloaders_anomaly_list is the number of categories of items
            dataloaders_list_relation_net.append(dataloaders_anomaly_list_relation_net)
        return dataloaders_list, dataloaders_list_relation_net

    return ("get_dataloaders_pretrain", get_dataloaders_pretrain)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
