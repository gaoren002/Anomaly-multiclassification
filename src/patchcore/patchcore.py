"""PatchCore and PatchCore detection methods."""
import copy
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torch.nn as nn
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
import wandb
LOGGER = logging.getLogger(__name__)
SHOT_LIST= ["000","001","002","003","004"]
class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.global_step = 0
    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            class_anomaly_subclass_number,
            writer,
            backbone_for_direct_classification=None,
            features = None,
            patchsize=3,
            patchstride=1,
            anomaly_score_num_nn=1,
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(False, 4),
            
            if_direct_classfication=False,
            if_mlp_classification=False,
            train_metric=False,
            train_contrastive_classification=False,
            relation_net=False,
            **kwargs,
    ):
        if features is not None:
            self.features=features
        self.writer=writer
        self.anomaly_class_number = class_anomaly_subclass_number
        self.backbone = backbone
        if isinstance(backbone, list):
            for i in range(len(backbone)):
                self.backbone[i] = self.backbone[i].to(device)
        else:
            self.backbone = backbone.to(device)

        if backbone_for_direct_classification is not None:
            self.backbone_for_direct_classification = backbone_for_direct_classification.to(device)

        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.criterion = nn.CrossEntropyLoss()
        self.residual_features_bank = None
        self.residual_features_bank_pretrain = None
        self.if_direct_classfication = if_direct_classfication
        self.train_metric = train_metric
        self.if_mlp_classification = if_mlp_classification
        self.train_contrastive_classification = train_contrastive_classification
        self.relation_net = relation_net
        if if_direct_classfication:  
            self.backbone_for_direct_classification.fc = nn.Linear(self.backbone_for_direct_classification.fc.in_features, class_anomaly_subclass_number)
            self.backbone_for_direct_classification.train()
            self.backbone_for_direct_classification = self.backbone_for_direct_classification.to(self.device)
            self.optimizer_for_direct_classification = torch.optim.Adam(self.backbone_for_direct_classification.parameters(), lr=0.00001)
        if train_metric:
            self.metric_model = patchcore.common.MyNetmetric().to(self.device)
            self.optimizer_metric = torch.optim.Adam(self.metric_model.parameters(), lr=1e-5)
        if if_mlp_classification: 
            self.classfication_mlp_model = patchcore.common.MyNetMLP(class_anomaly_subclass_number).to(self.device)
            self.optimizer_mlp = torch.optim.Adam(self.classfication_mlp_model.parameters(), lr=0.00001)
        if train_contrastive_classification:
            baseEncoder = patchcore.common.BaseEncoder().to(self.device)
            self.contrastive_model = patchcore.common.contrastive(baseEncoder).to(self.device)
            self.optimizers_contrastive = torch.optim.Adam(self.contrastive_model.parameters(), lr=0.0001)
            self.losses_q = [0 for _ in range(10)]  # losses_q[i] is the loss on step i
            self.corrects = [0 for _ in range(10)]
            self.update_lr = 0.001
            self.k=0
            self.flag=True
        if relation_net:
            self.relation_model = patchcore.common.RelationNet(class_anomaly_subclass_number).to(self.device)
            self.optimizers_relation = torch.optim.Adam(self.relation_model.parameters(), lr=0.0001)
        if isinstance(backbone, list):
            forward_modules = torch.nn.ModuleDict({})  # preprocess the features from resnet
            self.forward_modules_list = list()
            for i in range(len(backbone)): 
                feature_aggregator = patchcore.common.NetworkFeatureAggregator(
                self.backbone[i], self.layers_to_extract_from, self.device)

                feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
                forward_modules["feature_aggregator"] = feature_aggregator
                preprocessing = patchcore.common.Preprocessing(
                    feature_dimensions, pretrain_embed_dimension
                )
                forward_modules["preprocessing"] = preprocessing
                self.target_embed_dimension = target_embed_dimension
                preadapt_aggregator = patchcore.common.Aggregator(
                    target_dim=target_embed_dimension
                )

                _ = preadapt_aggregator.to(self.device)

                forward_modules["preadapt_aggregator"] = preadapt_aggregator
                self.forward_modules_list.append(forward_modules)

        else:
            self.forward_modules = torch.nn.ModuleDict({})  # preprocess the features from resnet
            feature_aggregator = patchcore.common.NetworkFeatureAggregator(
                self.backbone, self.layers_to_extract_from, self.device
            )
            feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
            self.forward_modules["feature_aggregator"] = feature_aggregator

            preprocessing = patchcore.common.Preprocessing(
                feature_dimensions, pretrain_embed_dimension
            )
            self.forward_modules["preprocessing"] = preprocessing
            self.target_embed_dimension = target_embed_dimension
            preadapt_aggregator = patchcore.common.Aggregator(
                target_dim=target_embed_dimension
            )

            _ = preadapt_aggregator.to(self.device)

            self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        def __embed(features):
            features = [features[layer] for layer in self.layers_to_extract_from]

            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ] 
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            ref_num_patches = patch_shapes[0]

            for i in range(1, len(features)):  
                _features = features[i]
                patch_dims = patch_shapes[i]

       
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )  # (2,14,14,1024,3,3)
                _features = _features.permute(0, -3, -2, -1, 1, 2)  # (2,1024,3,3,14,14)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])  # (2*1024*3*3,14,14)
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )  # (2*1024*3*3,1,28,28) the additional dim is used for interpolation
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )  # (2,1024,3,3,28,28)
                _features = _features.permute(0, -2, -1, 1, 2, 3)  # (2,28,28,1024,3,3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-3:]) for x in features]
            return features, patch_shapes
        if isinstance(self.backbone, list):
            features_list = list()
            for i in range(len(self.backbone)):
                _ = self.forward_modules_list[i]["feature_aggregator"].eval()
                with torch.no_grad():
                    features = self.forward_modules_list[i]["feature_aggregator"](images)

                features, patch_shapes = __embed(features)

                # As different feature backbones & patching provide differently
                # sized features, these are brought into the correct form here.
                features = self.forward_modules_list[i]["preprocessing"](features)  # (1568,2,1024)
                features = self.forward_modules_list[i]["preadapt_aggregator"](features)  # (1568,1024)
                features_list.append(features)
            features = torch.cat(features_list, dim=-1)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)
            features, patch_shapes = __embed(features)
            # As different feature backbones & patching provide differently
            # sized features, these are brought into the correct form here.
            features = self.forward_modules["preprocessing"](features)  # (1568,2,1024)
            features = self.forward_modules["preadapt_aggregator"](features)  # (1568,1024)
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        self.eval_mode()
        
        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image, provide_patch_shapes= True)

        features = []
        with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                feature, self.patch_shapes = _image_to_features(image)
                features.append(feature)

        features = np.concatenate(features, axis=0)
        self.features = self.featuresampler.run(features)  # (all_patches:17248, 1024)
        self.anomaly_scorer.fit(detection_features=[self.features])

    def pre_train(self, training_data, support_data, times=0,maml=False):

        # self.classfication_mlp_model = torch.load("./tmp.pth")
        self.global_step += 1
        result = 0
        if self.if_mlp_classification:
            flag = 0
            for i in range(1):
                for item in support_data:
                    anomaly = []
                    for j in range(len(item)):
                        tmp_anomaly = item[j]["anomaly_class"][0]
                        anomaly.append(tmp_anomaly)
                    self.anomaly_enum_for_mlp = dict()
                    
                    for i, item_ in enumerate(anomaly):
                        self.anomaly_enum_for_mlp[item_]=i
                    pass
                for i, item in enumerate(training_data):
                    images, anomaly, ori_imgs = [], [], []
                    for j in range(item['image'].shape[0]):
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly_class'][j]
                        ori_img = item["ori_img"][j]
                        ori_imgs.append(ori_img)
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
                    images = torch.stack(images, dim=0)
                    ori_imgs = torch.stack(ori_imgs, dim=0)
                    self.train_feature_mlp(images.to(self.device),ori_imgs.to(self.device), anomaly, pretrain=True)   

        if self.relation_net:
            print("分割线")
            for i in range(1):
                for i, item in enumerate(zip(support_data,training_data)): # training_data equavalent to query_data
                    support_item, query_item = item
                    result = self.train_relation_net(support_item, query_item,pretrain=True)  # the number of support is specified number of categories， query is selected randomly with no restriction on category
                    break
        if self.train_contrastive_classification:
            print("分割线")
            for i in range(1):
                for i, item in enumerate(zip(support_data,training_data)): # training_data equavalent to query_data， 每次取出batchsize个
                    support_item, query_item = item
                    result = self.train_contrastive_for_classification(support_item, query_item,pretrain=True,times=times,maml=maml)
                    break
        return result
    def train(self, test_data_for_training_anomaly_classfication,shot_num,running_times):
        images, anomaly, good_images = [], [], []
        self.anomaly_enum_for_mlp = dict()
        for i, item in enumerate(test_data_for_training_anomaly_classfication):
            for j in range(item['image'].shape[0]):
                for k in range(shot_num):
                    if SHOT_LIST[k] in item['image_name'][j] and item['anomaly'][j] != "good":  #TODO to fix the bug when the number of test images is over 1000
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly'][j]
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
        anomaly_num = len(set(anomaly))
        beishu = int(len(anomaly)/anomaly_num)
        for i in range(anomaly_num):
            self.anomaly_enum_for_mlp[anomaly[i*beishu]]=i
        # for k in range(len(anomaly)):
        #     anomaly[k] = label2index[anomaly[k]] 
        images = torch.stack(images, dim=0)

        if self.if_direct_classfication:
            anomaly_ = copy.deepcopy(anomaly)
            self.train_backbone_for_direct_classification(images.to(self.device), anomaly_) 
        if self.if_mlp_classification:
            anomaly_ = copy.deepcopy(anomaly)
            self.train_feature_mlp(images.to(self.device), None, anomaly_, pretrain=False) 
        if self.train_metric:
            for i, item in enumerate(test_data_for_training_anomaly_classfication):
                for j in range(item['image'].shape[0]):
                    if item['anomaly'][j] == "good":
                        tmp_image = item['image'][j]
                        good_images.append(tmp_image) 
            good_images = torch.stack(good_images, dim=0)
            self.train_feature_metric(images.to(self.device), anomaly, good_images.to(self.device),pretrain = False)
        if self.train_contrastive_classification:
            anomaly_ = copy.deepcopy(anomaly)
            self.train_contrastive_for_classification(images.to(self.device), anomaly_,pretrain=False,running_times=running_times)
        if self.relation_net:
            anomaly_ = copy.deepcopy(anomaly)
            self.train_relation_net(images.to(self.device), anomaly_,pretrain=False,running_times=running_times)

    def train_feature_mlp(self, images, ori_imgs, anomaly, pretrain=False,running_times=45):
        for k in range(len(anomaly)):
            anomaly[k] = self.anomaly_enum_for_mlp[anomaly[k]]
        """Infer score and mask for a batch of images."""
        if not pretrain:
            self.eval_mode()
            # self.classfication_mlp_model = torch.load("./tmp.pth")
            query_features = self._embed(images)
            query_features = np.asarray(query_features)

            image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
            query_nns = np.squeeze(query_nns, 1)
            nearest_neighbors_features = self.features[query_nns]  # (153664, 1024)
            residual_features = query_features - nearest_neighbors_features
            # residual_features = np.concatenate((residual_features,nearest_neighbors_features),-1)
            # residual_features = 2*residual_features + nearest_neighbors_features
            residual_features = residual_features.reshape(images.shape[0], -1)
            if self.residual_features_bank is None:
                self.residual_features_bank_ori = residual_features.reshape(len(anomaly), -1, residual_features.shape[1])
                self.residual_features_bank = np.mean(self.residual_features_bank_ori, axis=1)  
            residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
            
            labels = F.one_hot(torch.tensor(anomaly)).to(torch.float).to(self.device)
            # for params in self.classfication_mlp_model.parameters():
            #   params.requires_grad_(False)
            # self.classfication_mlp_model.linear2 = nn.Linear(128,4)
            # # nn.init.kaiming_normal_(self.classfication_mlp_model.linear2.weight)
            # self.classfication_mlp_model.to(self.device)
            # self.optimizer_mlp_2 = torch.optim.Adam(self.classfication_mlp_model.parameters(), lr=0.03)
            for i in range(running_times):
                results = self.classfication_mlp_model(residual_features)
                loss = self.criterion(results, labels)
                count = sum(results.argmax(-1)==labels.argmax(-1))
                self.optimizer_mlp.zero_grad()
                loss.backward()
                self.optimizer_mlp.step()
                print(count/results.shape[0])
        else:
            self.eval_mode()
        
            with torch.no_grad():
                ori_imgs = ori_imgs.to(torch.float).to(self.device)
                ori_feature, ori_patch_shapes = self._embed(ori_imgs, provide_patch_shapes= True)

            ori_feature = np.asarray(ori_feature)
            
            
            query_features = self._embed(images)
            query_features = np.asarray(query_features)

            image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
            query_nns = np.squeeze(query_nns, 1)
            nearest_neighbors_features = self.features[query_nns]  # (153664, 1024) when 220 images
            residual_features = query_features - ori_feature# nearest_neighbors_features
            residual_features = residual_features.reshape(images.shape[0], -1)
            
            if self.residual_features_bank_pretrain is None:  
                self.residual_features_bank_ori_pretrain = residual_features.reshape(len(anomaly), -1, residual_features.shape[1])
                self.residual_features_bank_pretrain = np.mean(self.residual_features_bank_ori_pretrain, axis=1)
            residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
            # labels = F.one_hot(torch.tensor(anomaly),num_classes=len(self.anomaly_enum_for_mlp)).to(torch.float).to(self.device)
            loss_sum = []
            # for j in range(residual_features.shape[0]):
            result= self.classfication_mlp_model(residual_features)
            loss = self.criterion(result, anomaly)
            # loss_sum.append(loss)
            self.optimizer_mlp.zero_grad()
            loss.backward()
            self.optimizer_mlp.step()
            print(loss)
            count = sum(result.argmax(-1) == labels.argmax(-1))
            print(count/result.shape[0])
            return result
    def train_relation_net(self,support_item, query_item_OR_anomaly, pretrain =False, support_num=1,running_times=25):
        if pretrain:
            self.eval_mode()
            query_item = query_item_OR_anomaly
            anomaly_support = []
            residual_features_relation_net = []
            for i,item in enumerate(support_item):
                images = item["image"].to(self.device)
                ori_imgs = item["ori_img"].to(self.device)
                anomaly_support.append(item["anomaly_class"])
                with torch.no_grad():
                    ori_imgs = ori_imgs.to(torch.float).to(self.device)
                    ori_feature, ori_patch_shapes = self._embed(ori_imgs, provide_patch_shapes= True)

                ori_feature = np.asarray(ori_feature)
                
                query_features = self._embed(images)
                query_features = np.asarray(query_features)
                
                # image_scores, query_distances, query_nns_support = self.anomaly_scorer.predict([query_features])
                # query_nns_support = np.squeeze(query_nns_support, 1)
                # nearest_neighbors_features_support = self.features[query_nns_support]
                
                residual_features = query_features - ori_feature# nearest_neighbors_feature
                residual_features = residual_features.reshape(images.shape[0], -1)

                residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
                residual_features_relation_net.append(residual_features) #                 The current first dimension is the number of categories, and the second dimension is the batch size (bs)
            residual_features_relation_net = torch.stack(residual_features_relation_net).permute(1,0,2) # （bs categories features' dimensions）
    #
            images_query = query_item["image"].to(self.device)
            ori_imgs_query = query_item["ori_img"].to(self.device)
            anomaly_query = query_item["anomaly_class"]
            with torch.no_grad():
                ori_imgs_query = ori_imgs_query.to(torch.float).to(self.device)
                ori_feature_query, ori_patch_shapes = self._embed(ori_imgs_query, provide_patch_shapes= True)

            ori_feature_query = np.asarray(ori_feature_query)
            query_features_query = self._embed(images_query)
            query_features_query = np.asarray(query_features_query) # the first query means input,  the second means query set
            
            # image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features_query])
            # query_nns = np.squeeze(query_nns, 1)
            # nearest_neighbors_features = self.features[query_nns]
            
            

        
            residual_features_query = query_features_query - ori_feature_query# nearest_neighbors_feature
            residual_features_query = residual_features_query.reshape(images_query.shape[0], -1)

            residual_features_query = torch.from_numpy(residual_features_query).to(self.device).requires_grad_(True)
            residual_features_query = residual_features_query.unsqueeze(1)    #（bs 1 features' dimensions）
            
            # random.shuffle(index)
            # invert_index = [0 for i in range(residual_features_relation_net.shape[1]) ]
            # for i,item in enumerate(index):
            #     invert_index [item] = i
            features = []
            for i in range(residual_features_relation_net.shape[0]):
                for j in range(residual_features_query.shape[0]):
                    for k in range(residual_features_relation_net.shape[1]):
                        item = torch.cat([residual_features_relation_net[i,k:k+1,:],residual_features_query[j]],dim = 1)
                        features.append(item)
            features = torch.cat(features,dim=0)
            labels = []
            for i in range(residual_features_relation_net.shape[0]):
                for k in range(residual_features_query.shape[0]):
                    for j in range(residual_features_relation_net.shape[1]):
                        if anomaly_support[j][i] == anomaly_query[k]:
                            labels.append(1)
                        else:
                            labels.append(0)
            labels = torch.tensor(np.array(labels)).to(torch.float).to(self.device)
            # 
            # anomaly_enum = [anomaly_support[i][0] for i in range(len(anomaly_support))]
            # random.shuffle(anomaly_enum) # label shuffle
            # anomaly_enum_dict = {anomaly_enum[i]:i for i in range(len(anomaly_enum))}
            # for i, item_ in enumerate(anomaly_reduced):
            #     anomaly_enum[item_]=i
            # for k in range(len(anomaly_query)):
            #     anomaly_query[k] = anomaly_enum[anomaly_query[k]]  # string to int
            # for i in range(len(anomaly_support)):
            #     for j in range(len(anomaly_support[i])):
            #         anomaly_support[i][j] = anomaly_enum[anomaly_support[i][j]]
            # labels_ori = [[anomaly_enum_dict[anomaly_query[i]]] for i in range(len(anomaly_query))]
            # labels = F.one_hot(torch.tensor(labels_ori),num_classes=self.anomaly_class_number).to(torch.float).to(self.device)


            # for j in range(residual_features.shape[0]):
            result1= self.relation_model(features)
            weights = labels.clone()
            for i in range(len(labels)):
                    if labels[i] ==1:
                        weights[i]=residual_features_relation_net.shape[1]-1
                    else:
                        weights[i]=1
            criterion = nn.BCELoss(weight=weights)
            loss = criterion(result1.squeeze(), labels)
            # loss2 = criterion(result2.squeeze(), labels)
            # loss = loss + loss2
            # loss_sum.append(loss)
            self.optimizers_relation.zero_grad()
            loss.backward()
            self.optimizers_relation.step()
            #print(loss)
            count = 0
            for i in range(len(labels)):
                if labels[i] ==1:
                    if result1[i][0]>0.5:
                        count += 1
                else:
                    if result1[i][0]<=0.5:
                        count += 1
            # result1 = result1.reshape(-1,residual_features_relation_net.shape[1]).argmax(-1)
            # labels = labels.reshape(-1,residual_features_relation_net.shape[1]).argmax(-1)
            # count = sum(result1 == labels)
            wandb.log({"running_accuracy":count/len(labels)})
            return count/len(labels)
        else:
            self.eval_mode()
           
            images = support_item.to(self.device)
            anomaly = query_item_OR_anomaly  
            query_features = self._embed(images)
            query_features = np.asarray(query_features) # the first support means input,  the second means support set
            image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
            query_nns = np.squeeze(query_nns, 1)
            nearest_neighbors_features = self.features[query_nns]  # (153664, 1024) when 220 images
            residual_features = query_features - nearest_neighbors_features
            residual_features = residual_features.reshape(images.shape[0], -1)
            residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
            

            # result1 = 0
            # classes_num = len(set(anomaly))
            # num_for_every_class = len(anomaly)//classes_num

            # for _ in range(running_times):
            #     for i in range(classes_num):
            #         for j in range(num_for_every_class):
            #             residual_features_query = []
                        
            #             residual_features_query.append(residual_features[i*num_for_every_class+j])
            #             residual_features_query = torch.stack(residual_features_query).detach()
            #             seen_index = i*num_for_every_class+j
            #             same_class_index = [i*num_for_every_class+l for l in range(num_for_every_class)]
            #             my_index = [x for x in same_class_index if x is not seen_index]
            #             for l in my_index:
            #                 residual_features_relation_net = []
            #                 for k in range(len(anomaly)):
            #                     if k not in same_class_index and (k+l+1)%num_for_every_class==0:
            #                         residual_features_relation_net.append(residual_features[k])
            #                     elif k==l:
            #                         residual_features_relation_net.append(residual_features[k])
            #                 # idx_shuffle = [i for i in range(len(residual_features_contrastive))]
            #                 # random.shuffle(idx_shuffle)
            #                 residual_features_relation_net = torch.stack(residual_features_relation_net)#[idx_shuffle]
            #                 residual_features_relation_net.unsqueeze_(0)
            #                 features = []
            #                 index = [i for i in range(residual_features_relation_net.shape[1]) ]
            #                 shape0 = residual_features_relation_net.shape[0] 
            #                 shape1 = residual_features_relation_net.shape[1]
            #                 shape2 = residual_features_query.shape[0]
            #                 for i in range(shape0):  # support_sum
            #                     for j in index: # class_num
            #                         for k in range(shape2): # query_num
            #                             item = torch.cat([residual_features_relation_net[i,j,:],residual_features_query[k]],dim = 0)
            #                             features.append(item)
            #                 features = torch.stack(features,dim=0).detach() 
                            



            #                 labels=F.one_hot(torch.tensor(i),classes_num).to(torch.float).unsqueeze_(0).to(self.device)
            #                 weights = labels.clone()
            #                 for i in range(len(labels)):
            #                         if labels[0][i] ==1:
            #                             weights[i]=residual_features_relation_net.shape[1]-1   
            #                         else:
            #                             weights[i]=1
            #                 criterion = nn.BCELoss()
                            
            #                 result1= self.relation_model(features) 
            #                 criterion = nn.CrossEntropyLoss()
                            
            #                 self.optimizers_relation.zero_grad()
            #                 loss = criterion(result1.transpose(0,1), labels)
            #                 loss.backward()
            #                 print(loss)
            #                 self.optimizers_relation.step()
            #                 count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
            #                 wandb.log({"running_accuracy_finetune":count/len(labels)})
            #                 self.writer.add_scalar("running_accuracy_finetune",count/len(labels),self.global_step)
                        
            # return result1

            residual_features_relation_net = []
            seen_index = []
            length = len(images)
            #perm = np.random.permutation(length) random select,deprecated
            anomaly_support = []
            for j in range(support_num):
                seen_support = []
                features = []
                anomaly_support_tmp = []
                for i in range(length):# perm:
                    if i not in seen_index:
                        if anomaly[i] not in seen_support:
                            seen_support.append(anomaly[i])
                            anomaly_support_tmp.append(anomaly[i])
                            features.append(residual_features[i])
                            seen_index.append(i)
                anomaly_support.append(anomaly_support_tmp)
                features = torch.stack(features).to(self.device)
                residual_features_relation_net.append(features)
            anomaly_query = []
            residual_features_query = []
            for i in range(length):
                if i not in seen_index:
                    anomaly_query.append(anomaly[i])
                    residual_features_query.append(residual_features[i])
            residual_features_relation_net = torch.stack(residual_features_relation_net)  # supportnum classes features' dimensions
            if len(residual_features_query)==0:
                return 0
            residual_features_query = torch.stack(residual_features_query)
            features = []
            index = [i for i in range(residual_features_relation_net.shape[1]) ]
            shape0 = residual_features_relation_net.shape[0] # batchsize or support_num
            shape1 = residual_features_relation_net.shape[1] # the number of categories  4
            shape2 = residual_features_query.shape[0]
            for i in range(shape0):  # support_sum
                for j in index: # class_num
                    for k in range(shape2): # query_num
                        item = torch.cat([residual_features_relation_net[i,j,:],residual_features_query[k]],dim = 0)
                        features.append(item)
            features = torch.stack(features,dim=0).detach() # concat




            labels = []
            for i in range(shape0): #support_num
                for j in range(shape1): # class_num
                    for k in range(shape2): # query_num
                        if anomaly_support[i][j] == anomaly_query[k]:
                            labels.append(1)
                        else:
                            labels.append(0)
            labels = torch.tensor(np.array(labels)).to(torch.float).to(self.device).detach()
            weights = labels.clone()
            for i in range(len(labels)):
                    if labels[i] ==1:
                        weights[i]=residual_features_relation_net.shape[1]-1   
                    else:
                        weights[i]=1
            criterion = nn.BCELoss(weight=weights)
            for i in range(running_times):
                result1= self.relation_model(features) # The former randomly selects 10 Qs, while the latter is ten sets, used as K.
                loss = criterion(result1.squeeze(), labels)
                self.optimizers_relation.zero_grad()
                loss.backward()
                print(loss)
                self.optimizers_relation.step()
                
                count = 0
                for i in range(len(labels)):
                    if labels[i] ==1:
                        if result1[i][0]>0.5:
                            count += 1
                    else:
                        if result1[i][0]<=0.5:
                            count += 1
            # count = sum(result.argmax(-1) == labels.argmax(-1))
            wandb.log({"running_accuracy_finetune":count/len(labels)})  #TODO tensorboard 
            return count/len(labels)


    def train_backbone_for_direct_classification(self, images, anomaly):
        """Infer score and mask for a batch of images."""
        for k in range(len(anomaly)):
            anomaly[k] = self.anomaly_enum_for_mlp[anomaly[k]]
        results = self.backbone_for_direct_classification(images)
        labels = F.one_hot(torch.tensor(anomaly)).to(torch.float).to(self.device)
        for i in range(100):
            for j in range(images.shape[0]):
                results = self.backbone_for_direct_classification(images[j].unsqueeze(0)).squeeze(0)

                loss = self.criterion(results, labels[j])
                if i % 10 == 0:
                    print(results)
                    print(loss)
                self.optimizer_for_direct_classification.zero_grad()
                loss.backward()
                self.optimizer_for_direct_classification.step()
        return results

    def train_feature_metric(self, images, anomaly, good_images, pretrain =False): # deprecated
        if pretrain is True:
            if self.residual_features_bank_pretrain is None:
                self.eval_mode()
                query_features = self._embed(images)
                query_features = np.asarray(query_features)

                image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
                query_nns = np.squeeze(query_nns, 1)
                nearest_neighbors_features = self.features[query_nns]
                residual_features = query_features - nearest_neighbors_features
                residual_features = residual_features.reshape(images.shape[0], -1)
                self.residual_features_bank_ori_pretrain = residual_features.reshape(len(anomaly), -1, residual_features.shape[1])
                self.residual_features_bank_pretrain = np.mean(self.residual_features_bank_ori_pretrain, axis=1)
            torch.cuda.empty_cache()
            residual_features = torch.from_numpy(self.residual_features_bank_ori_pretrain).reshape(-1, 28, 28, 1024).to(self.device).requires_grad_(True)
            residual_features = residual_features.permute(0, 3, 1, 2)
            
            metric_model = self.metric_model.to(self.device)
            anomaly_a = torch.tensor(anomaly).unsqueeze(0)
            anomaly_b = torch.tensor(anomaly).unsqueeze(1)
            labels = (anomaly_a == anomaly_b).to(torch.long).to(self.device)
            index = (torch.ones_like(labels) - labels).to(torch.bool)
            torch.autograd.set_detect_anomaly(True)
            # self.simple_proj = nn.Linear(784*1024, 1024).requires_grad_(False).to(self.device)
            for i in range(400):
                # results_embeddings = self.simple_proj(residual_features)
                results_embeddings = metric_model(residual_features)
                # results_embeddings = results_embeddings / torch.norm(results_embeddings, dim=-1, keepdim=True)
                print(results_embeddings.grad)
                a = results_embeddings[None, ...]
                b = results_embeddings.unsqueeze(1)
                c = a - b
                c = c[index]
                distances = torch.sum(c**2, dim=-1)
                distances_far = 1 / distances.mean()
                loss = distances_far
                self.optimizer_metric.zero_grad()
                loss.backward()
                print(loss, distances_far)
                self.optimizer_metric.step()
        else:
            if self.residual_features_bank is None:
                self.eval_mode()
                query_features = self._embed(images)
                query_features = np.asarray(query_features)

                image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
                query_nns = np.squeeze(query_nns, 1)
                nearest_neighbors_features = self.features[query_nns]
                residual_features = query_features - nearest_neighbors_features
                residual_features = residual_features.reshape(images.shape[0], -1)
                self.residual_features_bank_ori = residual_features.reshape(len(anomaly), -1, residual_features.shape[1])
                self.residual_features_bank = np.mean(self.residual_features_bank_ori, axis=1)
                # only leave a prototype for each anomaly class
            torch.cuda.empty_cache()
            residual_features = torch.from_numpy(self.residual_features_bank_ori).reshape(-1, 28, 28, 1024).to(self.device).requires_grad_(True)
            residual_features = residual_features.permute(0, 3, 1, 2)
            # self.metric_model.fc = nn.Linear(28*28,4)
            metric_model = self.metric_model.to(self.device)
            anomaly_a = torch.tensor(anomaly).unsqueeze(0)
            anomaly_b = torch.tensor(anomaly).unsqueeze(1)
            labels = (anomaly_a == anomaly_b).to(torch.long).to(self.device)
            index = (torch.ones_like(labels) - labels).to(torch.bool)
            torch.autograd.set_detect_anomaly(True)
            # self.simple_proj = nn.Linear(784*1024, 1024).requires_grad_(False).to(self.device)
            for i in range(400):
                # results_embeddings = self.simple_proj(residual_features)
                results_embeddings = metric_model(residual_features)
                # results_embeddings = results_embeddings / torch.norm(results_embeddings, dim=-1, keepdim=True)
                print(results_embeddings.grad)
                a = results_embeddings[None, ...]
                b = results_embeddings.unsqueeze(1)
                c = a - b
                c = c[index]
                distances = torch.sum(c**2, dim=-1)
                distances_far = 1 / distances.mean()
                loss = distances_far
                self.optimizer_metric.zero_grad()
                loss.backward()
                print(loss, distances_far)
                self.optimizer_metric.step()
        return results_embeddings

    def train_contrastive_for_classification(self, support_item_OR_images, query_item_OR_anomaly, pretrain =False, running_times=15,times=-1,maml=False):
        if pretrain is True:
            self.eval_mode()
            query_item = query_item_OR_anomaly
            anomaly_support = []
            residual_features_contrastive = []
            support_item = support_item_OR_images
            for i,item in enumerate(support_item):
                images = item["image"].to(self.device)
                ori_imgs = item["ori_img"].to(self.device)
                anomaly_support.append(item["anomaly_class"])
                with torch.no_grad():
                    ori_imgs = ori_imgs.to(torch.float).to(self.device)
                    ori_feature, ori_patch_shapes = self._embed(ori_imgs, provide_patch_shapes= True)
                ori_feature = np.asarray(ori_feature)
                
                query_features = self._embed(images)
                query_features = np.asarray(query_features)

                residual_features = query_features - ori_feature
                residual_features = residual_features.reshape(images.shape[0], -1)

                residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
                residual_features_contrastive.append(residual_features)
            residual_features_contrastive = torch.stack(residual_features_contrastive).permute(1,0,2)
            images_query = query_item["image"].to(self.device)
            ori_imgs_query = query_item["ori_img"].to(self.device)
            anomaly_query = query_item["anomaly_class"]
            with torch.no_grad():
                ori_imgs_query = ori_imgs_query.to(torch.float).to(self.device)
                ori_feature_query, ori_patch_shapes = self._embed(ori_imgs_query, provide_patch_shapes= True)
            ori_feature_query = np.asarray(ori_feature_query)
            query_features_query = self._embed(images_query)
            query_features_query = np.asarray(query_features_query) # the first query means input,  the second means query set
            residual_features_query = query_features_query - ori_feature_query# nearest_neighbors_feature
            residual_features_query = residual_features_query.reshape(images_query.shape[0], -1)
            residual_features_query = torch.from_numpy(residual_features_query).to(self.device).requires_grad_(True)
            
            shape0 = residual_features_contrastive.shape[0]
            shape1 = residual_features_contrastive.shape[1]
            labels = []
            classes = []
            for j in range(shape1):
                classes.append(anomaly_support[j][0])
            for i in range(residual_features_query.shape[0]):
                index = classes.index(anomaly_query[i])
                labels.append(index)
            labels = torch.tensor(np.array(labels)).to(self.device)
            labels = labels.repeat((shape0,1)).reshape(-1)
            labels = F.one_hot(labels,len(classes)).to(torch.float32)
            if not maml:
            
                result1= self.contrastive_model(residual_features_query, residual_features_contrastive) # The former randomly selects n q, for example (10,82816), while the latter is ten sets, each containing class_num items, for example (10, 20, 802816), used as K, with the shape of query in labels taking precedence.
                # result 100*20
                criterion = nn.CrossEntropyLoss()
                loss = criterion(result1, labels)
                self.optimizers_contrastive.zero_grad()
                loss.backward()
                self.optimizers_contrastive.step()
               

                count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
                wandb.log({"running_accuracy":count/len(labels)})
                self.writer.add_scalar("running_accuracy",count/len(labels),self.global_step)
                return count/len(labels)
            else:
                criterion = nn.CrossEntropyLoss()
                if times==-1:
                    result1= self.contrastive_model(residual_features_query, residual_features_contrastive,vars=None) # The former randomly selects n q, for example (10,82816), while the latter is ten sets, each containing class_num items, for example (10, 20, 802816), used as K, with the shape of query in labels taking precedence.
                    # result 100*20
                    loss = criterion(result1, labels)
                    grad = torch.autograd.grad(loss, self.contrastive_model.encoder_q.parameters(), allow_unused=True)
                    self.fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else None, zip(grad, self.contrastive_model.encoder_q.parameters())))
                    with torch.no_grad():
                        # [setsz, nway]
                        result1 = self.contrastive_model(residual_features_query, residual_features_contrastive,vars=self.fast_weights)
                        loss_q = criterion(result1, labels)
                        self.losses_q[0] += loss_q
                        # [setsz]
                        count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
                        wandb.log({"running_accuracy":count/len(labels)})
                        self.writer.add_scalar("running_accuracy",count/len(labels),self.global_step)
                        return count/len(labels)
                if self.flag: # support
                    result1 = self.contrastive_model(residual_features_query, residual_features_contrastive,vars=self.fast_weights)
                    loss_q = criterion(result1, labels)
                    grad = torch.autograd.grad(loss_q, self.contrastive_model.encoder_q.parameters(), allow_unused=True)
                    self.fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else None, zip(grad, self.contrastive_model.encoder_q.parameters())))
                    self.flag = False
                    count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
                    wandb.log({"running_accuracy":count/len(labels)})
                    self.writer.add_scalar("running_accuracy",count/len(labels),self.global_step)
                    return count/len(labels)
                else:  # query
                    result1 = self.contrastive_model(residual_features_query, residual_features_contrastive,vars=self.fast_weights)
                    loss_q = criterion(result1, labels)
                   
                    self.losses_q[(times//2)%10] += loss_q
                    count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
                    wandb.log({"running_accuracy":count/len(labels)})
                    self.writer.add_scalar("running_accuracy",count/len(labels),self.global_step)
                    
                    if self.k%10==9:
                        loss_q = self.losses_q[-1]
                        # [setsz]

                        self.optimizers_contrastive.zero_grad()
                        loss_q.backward()
                        self.optimizers_contrastive.step()
                        self.losses_q = [0 for _ in range(10)]
                        count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
                        wandb.log({"running_accuracy":count/len(labels)})
                        self.writer.add_scalar("running_accuracy",count/len(labels),self.global_step)
                    self.k+=1
                    self.flag = True
                    return count/len(labels)
        else:
            
            self.eval_mode()
            images = support_item_OR_images.to(self.device)
            anomaly = query_item_OR_anomaly  
            query_features = self._embed(images)
            query_features = np.asarray(query_features) # the first support means input,  the second means support set
            image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
            query_nns = np.squeeze(query_nns, 1)
            nearest_neighbors_features = self.features[query_nns]  # (153664, 1024) when 220 images
            residual_features = query_features - nearest_neighbors_features
            # residual_features = query_features
            residual_features = residual_features.reshape(images.shape[0], -1)
            residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
            




            result1 = 0
            classes_num = len(set(anomaly))
            num_for_every_class = len(anomaly)//classes_num
            torch.autograd.set_detect_anomaly(True)

            for _ in range(running_times):
                for i in range(classes_num):
                    for j in range(num_for_every_class):
                        residual_features_query = []
                        
                        residual_features_query.append(residual_features[i*num_for_every_class+j])
                        residual_features_query = torch.stack(residual_features_query).detach()
                        seen_index = i*num_for_every_class+j
                        same_class_index = [i*num_for_every_class+l for l in range(num_for_every_class)]
                        my_index = [x for x in same_class_index if x is not seen_index]
                        for l in my_index:
                            residual_features_contrastive = []
                            for k in range(len(anomaly)):
                                if k not in same_class_index and (k+l+1)%num_for_every_class==0:
                                    residual_features_contrastive.append(residual_features[k])
                                elif k==l:
                                    residual_features_contrastive.append(residual_features[k])
                            # idx_shuffle = [i for i in range(len(residual_features_contrastive))]
                            # random.shuffle(idx_shuffle)
                            residual_features_contrastive = torch.stack(residual_features_contrastive)#[idx_shuffle]
                            residual_features_contrastive.unsqueeze_(0)
                            labels=F.one_hot(torch.tensor(i),classes_num).to(torch.float).unsqueeze_(0).to(self.device)
                            
                            result1= self.contrastive_model(residual_features_query, residual_features_contrastive)# The former randomly selects 10 Q, while the latter is ten sets, used as K.
                            criterion = nn.CrossEntropyLoss()
                            
                            self.optimizers_contrastive.zero_grad()
                            loss = criterion(result1, labels)
                            loss.backward()
                            print(loss)
                            self.optimizers_contrastive.step()
                            count = torch.sum(result1.argmax(dim=1)==labels.argmax(dim=1))
                            wandb.log({"running_accuracy_finetune":count/len(labels)})
                            self.writer.add_scalar("running_accuracy_finetune",count/len(labels),self.global_step)
                        
            return result1

    def predict(self, data,shot_num): 
        if isinstance(data, torch.utils.data.DataLoader):
            if self.if_direct_classfication:
                self.predict_for_direct_classfication(data, shot_num=shot_num) 
            if self.if_mlp_classification:
                self.predict_for_classification_mlp(data, shot_num=shot_num)  
            if self.relation_net:
                self.predict_for_relation_net(data, shot_num=shot_num)
            if self.train_contrastive_classification:
                self.predict_contrastive(data, shot_num=shot_num)
            if self.train_metric:
                #TODO: metric learning
                self.predict_for_metric(data)  
            return self._predict_dataloader(data)  # the original prediction of patchcore
        return self._predict(data)  # the original prediction of patchcore

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.eval_mode()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        self.eval_mode()
        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            image_scores, query_distances, query_nns = self.anomaly_scorer.predict([features])

            patch_scores = image_scores
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )

            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            return [score for score in image_scores], [mask for mask in masks]

    def predict_contrastive(self,dataloader,last=False,last_pretrain=False,shot_num=2):       
        with tqdm.tqdm(dataloader, desc="Classification for contrastive...", leave=False) as data_iterator:
            images, anomaly = [], []
            images_support, anomaly_support = [], []
            for i, item in enumerate(data_iterator):
                for j in range(item['image'].shape[0]):
                    flag = False
                    for k in range(shot_num):
                        if SHOT_LIST[k] in item['image_name'][j]:
                            flag = True
                    if not flag and item['anomaly'][
                        j] != "good":
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly'][j]
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
                    elif flag and item['anomaly'][
                        j] != "good":
                        tmp_image_support = item['image'][j]
                        tmp_anomaly_support = item['anomaly'][j]
                        images_support.append(tmp_image_support)
                        anomaly_support.append(tmp_anomaly_support)
            # for k in range(len(anomaly)):
            #     anomaly[k] = label2index[anomaly[k]]
            images = torch.stack(images, dim=0)
            images_support = torch.stack(images_support, dim=0)
            count = [0 for i in range(len(set(anomaly)))]
            features_tsne_before = []
            features_tsne_after = []
            if images.shape[0]>10:
                num = images.shape[0]//10
                yu = images.shape[0]%10
                for l in range(num):
                    result,anomaly_name,features,features_support,contrastive_features, contrastive_features_support = self._predict_contrastive(images[l*10:(l+1)*10], anomaly[l*10:(l+1)*10], images_support, anomaly_support)
                    count = [result[i]+count[i] for i in range(len(count))]
                    features_tsne_before.append(features)
                    features_tsne_after.append(contrastive_features)
                if yu !=0:
                    result,anomaly_name,features,features_support,contrastive_features, contrastive_features_support = self._predict_contrastive(images[num*10:], anomaly[num*10:], images_support, anomaly_support)
                    count = [result[i]+count[i] for i in range(len(count))]
                    features_tsne_before.append(features)
                    features_tsne_after.append(contrastive_features)
            else:
                result,anomaly_name,features,features_support,contrastive_features, contrastive_features_support = self._predict_contrastive(images, anomaly, images_support, anomaly_support)
                count = [result[i]+count[i] for i in range(len(count))]
                features_tsne_before.append(features)
                features_tsne_after.append(contrastive_features)


            features_tsne_before.append(features_support)
            features_tsne_after.append(contrastive_features_support)
            X_before = np.array(torch.cat(features_tsne_before,dim=0).detach().cpu())
            X_after = np.array(torch.cat(features_tsne_after,dim=0).detach().cpu())
            tsne = TSNE(n_components=2, init='pca', random_state=0,perplexity=20)
            X_tsne = tsne.fit_transform(X_before)
            X_tsne2 = tsne.fit_transform(X_after)
            unique_elements = list(set(anomaly))
            mapping = {element: idx for idx, element in enumerate(unique_elements, 1)}
            numeric_list = [mapping[anomaly_] for anomaly_ in anomaly]
            numeric_list2 = [mapping[anomaly_] for anomaly_ in  anomaly_name ]
            numeric_list.extend(numeric_list2)
            y = np.array(numeric_list)
            plt.clf()
            plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y)
            plt.savefig('my_figure.png')
            plt.clf()
            plt.scatter(X_tsne2[:,0],X_tsne2[:,1],c=y)
            plt.savefig('my_figure2.png')




            binary_accuracy = sum(count)/images.shape[0]
            accuracy_respectively = [count[i]/anomaly.count(anomaly_name[i]) for i in range(len(count))] 
            print("classification_contrastive:",binary_accuracy)
            print("respectively classfification:",accuracy_respectively)
            print("respectively class",anomaly_name)
            if last:
                wandb.log({#"binary classification:":binary_accuracy,
                    "last_classification:": binary_accuracy})
                self.writer.add_scalar("last_classification",binary_accuracy,self.global_step)
            elif last_pretrain:
                wandb.log({#"binary classification:":binary_accuracy,
                    "last_classification_pretrain:": binary_accuracy})
                self.writer.add_scalar("last_classification_pretrain",binary_accuracy,self.global_step)
                wandb.log({#"binary classification:":binary_accuracy,
                    "classification:": binary_accuracy})
                self.writer.add_scalar("classification",binary_accuracy,self.global_step)
            else:
                wandb.log({#"binary classification:":binary_accuracy,
                    "classification:": binary_accuracy})
                self.writer.add_scalar("classification",binary_accuracy,self.global_step)
            return binary_accuracy
        
    def _predict_contrastive(self, images, anomalys, images_support, anomalys_support):
        self.eval_mode()
        self.contrastive_model.eval()
        anomalys_support_new = []
        images_support_new = []
        support_num = len(anomalys_support) // len(set(anomalys_support))
        for i in range(len(images_support)):
            if anomalys_support[i] not in anomalys_support_new: 
                images_support_new.append(images_support[i]/support_num)
                anomalys_support_new.append(anomalys_support[i])
            else:
                idx = anomalys_support_new.index(anomalys_support[i])
                images_support_new[idx] += images_support[i]/support_num
        images_support_new = torch.stack(images_support_new,dim=0)
        query_features_support = self._embed(images_support_new.to(self.device))
        query_features_support = np.asarray(query_features_support)
        image_scores, query_distances, query_nns_support = self.anomaly_scorer.predict([query_features_support])
        query_nns_support = np.squeeze(query_nns_support, 1)
        nearest_neighbors_features_support = self.features[query_nns_support]
        residual_features_support = query_features_support - nearest_neighbors_features_support
        residual_features_support = residual_features_support.reshape(images_support_new.shape[0], -1)
        residual_features_support = torch.from_numpy(residual_features_support).to(self.device)
        # (n 85, c 802816)
        query_features = self._embed(images.to(self.device))
        query_features = np.asarray(query_features)

        image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
        query_nns = np.squeeze(query_nns, 1)
        nearest_neighbors_features = self.features[query_nns]
        residual_features_query = query_features - nearest_neighbors_features
        # (n 4, c 802816)
        residual_features_query = residual_features_query.reshape(images.shape[0], -1)
        residual_features_query = torch.from_numpy(residual_features_query).to(self.device)
        features = []
        anomalys_last = []
        
        shape_support = residual_features_support.shape[0]
        shape_query = residual_features_query.shape[0]
        # residual_features_query =residual_features_query.repeat((shape1,1))

        labels = []                   

        for i in range(shape_query):
            index = anomalys_support_new.index(anomalys[i])
            labels.append(index)
        
        labels = torch.tensor(np.array(labels)).to(self.device)
        labels = F.one_hot(labels,len(anomalys_support_new)).to(torch.float32)
        residual_features_support.unsqueeze_(0)
        results = []


        for i in range(residual_features_query.shape[0]):
            results.append(self.contrastive_model(residual_features_query[i:i+1], residual_features_support)) # The former randomly selects 10 Q, while the latter is ten sets, used as K.      
        results = torch.cat(results,dim=0)

        contrastive_features_query = self.contrastive_model.encoder_q(residual_features_query)
        contrastive_features_support = self.contrastive_model.encoder_q(residual_features_support[0])
        labels_index = labels.argmax(dim=1)
        results = results.argmax(dim=1)
        count = [0 for i in range(labels.shape[1])] # count is an array for every defect class
        for i,label in enumerate(labels_index):
            if results[i]==label:
                for j in range(len(count)):
                    if label==j:
                        count[j]+=1
        
        return count, anomalys_support_new, residual_features_query, residual_features_support[0], contrastive_features_query, contrastive_features_support  # return count for classification accuracy
    # anomalys_support_new for a dict of index to anomaly
    def predict_for_classification_mlp(self, dataloader,last=False,last_pretrain=False, shot_num=2):

        # label2index = {"bent": 0, "color": 1, "flip": 2, "scratch": 3}
        #label2index = {"bent": 0, "color": 1, "flip": 2, "scratch": 3}
        images, anomaly = [], []
        with tqdm.tqdm(dataloader, desc="Classification by mlp...", leave=False) as data_iterator:
            for i, item in enumerate(data_iterator):
                for j in range(item['image'].shape[0]):
                    flag = False
                    for k in range(shot_num):
                        if SHOT_LIST[k] in item['image_name'][j]:
                            flag = True
                    if not flag and item['anomaly'][
                        j] != "good":
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly'][j]
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
            # for k in range(len(anomaly)):
            #     anomaly[k] = label2index[anomaly[k]]
            images = torch.stack(images, dim=0)
            anomaly_num = len(set(anomaly))
            count = [0 for i in range(anomaly_num)]
            
            for k in range(len(anomaly)):
                anomaly[k] = self.anomaly_enum_for_mlp[anomaly[k]]
            if images.shape[0]>10:
                num = images.shape[0]//10
                yu = images.shape[0]%10
                for l in range(num):
                    result = self._predict_for_classification_mlp(images[l*10:(l+1)*10], anomaly[l*10:(l+1)*10],anomaly_num)
                    count = [result[i]+count[i] for i in range(len(count))]
                if yu !=0:
                    result = self._predict_for_classification_mlp(images[num*10:], anomaly[num*10:],anomaly_num)
                    count = [result[i]+count[i] for i in range(len(count))]
            else:
                result = self._predict_for_classification_mlp(images, anomaly,anomaly_num)
                count = [result[i]+count[i] for i in range(len(count))]
            
            binary_accuracy = sum(count)/images.shape[0]
            accuracy_respectively = [count[i]/anomaly.count(i) for i in range(len(count))]
            print("classification_mlp:",binary_accuracy)
            print("respectively classfification mlp:",accuracy_respectively)
            print("respectively class",self.anomaly_enum_for_mlp)
            if last:
                wandb.log({#"binary classification:":binary_accuracy,
                    "last_classification:": binary_accuracy})
                self.writer.add_scalar("last_classification",binary_accuracy,self.global_step)
            elif last_pretrain:
                wandb.log({#"binary classification:":binary_accuracy,
                    "last_classification_pretrain:": binary_accuracy})
                self.writer.add_scalar("last_classification_pretrain",binary_accuracy,self.global_step)
                wandb.log({#"binary classification:":binary_accuracy,
                    "classification:": binary_accuracy})
                self.writer.add_scalar("classification",binary_accuracy,self.global_step)
            else:
                wandb.log({#"binary classification:":binary_accuracy,
                    "classification:": binary_accuracy})
                self.writer.add_scalar("classification",binary_accuracy,self.global_step)
            return 
            

    def _predict_for_classification_mlp(self, images, anomaly, anomaly_num):
        self.eval_mode()
        self.classfication_mlp_model.eval()
        # self.index2label = {0: "bent", 1: "color", 2: "flip", 3: "scratch"}
        query_features = self._embed(images.to(self.device))
        query_features = np.asarray(query_features)

        image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
        query_nns = np.squeeze(query_nns, 1)
        nearest_neighbors_features = self.features[query_nns]
        residual_features = query_features - nearest_neighbors_features
        # residual_features = np.concatenate((residual_features,nearest_neighbors_features),-1)
        # residual_features = 2*residual_features + nearest_neighbors_features
        residual_features = residual_features.reshape(images.shape[0], -1)
        residual_features = torch.from_numpy(residual_features).to(self.device).requires_grad_(True)
        results = self.classfication_mlp_model(residual_features)
        labels = F.one_hot(torch.tensor(anomaly),anomaly_num).to(torch.float).to(self.device)
        count = [0 for i in range(labels.shape[1])]
        labels_index = labels.argmax(dim=1)
        results = results.argmax(dim=1)
         # count is an array for every defect class
        for i,label in enumerate(labels_index):
            if results[i]==label:
                for j in range(len(count)):
                    if label==j:
                        count[j]+=1
        return count
    def predict_for_relation_net(self,dataloader,last=False,last_pretrain=False, shot_num=2):
        # label2index = {"bent": 0, "color": 1, "flip": 2, "scratch": 3}
        images, anomaly = [], []
        images_support, anomaly_support = [], []
        with tqdm.tqdm(dataloader, desc="Classification for relationNet...", leave=False) as data_iterator:
            for i, item in enumerate(data_iterator):
                for j in range(item['image'].shape[0]):
                    flag = False
                    for k in range(shot_num):
                        if SHOT_LIST[k] in item['image_name'][j]:
                            flag = True
                    if not flag and item['anomaly'][
                        j] != "good":
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly'][j]
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
                    elif flag and item['anomaly'][
                        j] != "good":
                        tmp_image_support = item['image'][j]
                        tmp_anomaly_support = item['anomaly'][j]
                        images_support.append(tmp_image_support)
                        anomaly_support.append(tmp_anomaly_support)
            # for k in range(len(anomaly)):
            #     anomaly[k] = label2index[anomaly[k]]
            images = torch.stack(images, dim=0)
            images_support = torch.stack(images_support, dim=0)
            anomaly_num = len(set(anomaly))
            count1 = 0
            count2 = [0 for i in range(len(set(anomaly)))]
            if images.shape[0]>10:
                num = images.shape[0]//10
                yu = images.shape[0]%10
                for l in range(num):
                    result_binary, result_classification,anomaly_name = self._predict_for_relation_net(images[l*10:(l+1)*10], anomaly[l*10:(l+1)*10], images_support, anomaly_support, anomaly_num)
                    count1 += result_binary
                    count2 = [result_classification[i]+count2[i] for i in range(len(count2))]
                if yu!=0:
                    result_binary, result_classification,anomaly_name = self._predict_for_relation_net(images[num*10:], anomaly[num*10:], images_support, anomaly_support, anomaly_num)
                    count1 += result_binary
                    count2 = [result_classification[i]+count2[i] for i in range(len(count2))]
            else:
                result_binary, result_classification, anomaly_name = self._predict_for_relation_net(images, anomaly, images_support, anomaly_support, anomaly_num)
                count1 += result_binary
                count2 = [result_classification[i]+count2[i] for i in range(len(count2))]
            binary_accuracy = count1 / images.shape[0]
            print("binary classification:",binary_accuracy)
            accuracy_respectively = [count2[i]/anomaly.count(anomaly_name[i]) for i in range(len(count2))]
            print("respectively classfification:",accuracy_respectively)
            print("respectively class",anomaly_name)
            classification_accuracy = sum(count2) / images.shape[0]
            print("classification_relationnet:", classification_accuracy)
            if last:
                wandb.log({"binary classification:":binary_accuracy,
                    "last_classification:": classification_accuracy})
                self.writer.add_scalar("last_classification",classification_accuracy,self.global_step)
                self.writer.add_scalar("binary classification:",binary_accuracy,self.global_step)
            elif last_pretrain:
                wandb.log({"binary classification:":binary_accuracy,
                    "last_classification_pretrain:": classification_accuracy,
                        "classification:": classification_accuracy
                    })
                self.writer.add_scalar("last_classification_pretrain",classification_accuracy,self.global_step)
                self.writer.add_scalar("classification",classification_accuracy,self.global_step)
                self.writer.add_scalar("binary classification:",binary_accuracy,self.global_step)
            else:
                wandb.log({"binary classification:":binary_accuracy,
                    "classification:": classification_accuracy})
                self.writer.add_scalar("classification",classification_accuracy,self.global_step)
                self.writer.add_scalar("binary classification:",binary_accuracy,self.global_step)
            
            return classification_accuracy
    def _predict_for_relation_net(self, images, anomalys, images_support, anomalys_support,anomaly_num):
        self.eval_mode()
        self.relation_model.eval()
        anomalys_support_new = []
        images_support_new = []
        support_num = len(anomalys_support) // len(set(anomalys_support))
        for i in range(len(images_support)):
            if anomalys_support[i] not in anomalys_support_new:  
                images_support_new.append(images_support[i]/support_num)
                anomalys_support_new.append(anomalys_support[i])
            else:
                idx = anomalys_support_new.index(anomalys_support[i])
                images_support_new[idx] += images_support[i]/support_num
        images_support_new = torch.stack(images_support_new,dim=0)
        query_features_support = self._embed(images_support_new.to(self.device))
        query_features_support = np.asarray(query_features_support)
        image_scores, query_distances, query_nns_support = self.anomaly_scorer.predict([query_features_support])
        query_nns_support = np.squeeze(query_nns_support, 1)
        nearest_neighbors_features_support = self.features[query_nns_support]
        residual_features_support = query_features_support - nearest_neighbors_features_support
        residual_features_support = residual_features_support.reshape(images_support_new.shape[0], -1)
        residual_features_support = torch.from_numpy(residual_features_support).to(self.device)
        
        query_features = self._embed(images.to(self.device))
        query_features = np.asarray(query_features)

        image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
        query_nns = np.squeeze(query_nns, 1)
        nearest_neighbors_features = self.features[query_nns]
        residual_features_query = query_features - nearest_neighbors_features
        # residual_features = np.concatenate((residual_features,nearest_neighbors_features),-1)
        # residual_features = 2*residual_features + nearest_neighbors_features
        residual_features_query = residual_features_query.reshape(images.shape[0], -1)
        residual_features_query = torch.from_numpy(residual_features_query).to(self.device).requires_grad_(True)
        features = []
        anomalys_last = []
        for j in range(residual_features_query.shape[0]):
            for i in range(residual_features_support.shape[0]):
                item = torch.cat([residual_features_support[i],residual_features_query[j]])
                features.append(item.detach_().cpu().numpy())
                if anomalys_support_new[i]==anomalys[j]:
                    anomalys_last.append(1)
                else:
                    anomalys_last.append(0)
        features = torch.tensor(np.array(features))
        features = features.reshape(features.shape[0],-1).to(self.device)
        # anomalys = [anomalys_suport_new.index(anomalys[i]) for i in range(len(anomalys))]
        # labels = F.one_hot(torch.tensor(anomalys)).to(torch.float).to(self.device)
        labels = torch.tensor(np.array(anomalys_last)).to(torch.float).to(self.device)
        count1 = 0
        result1 = self.relation_model(features)
        for i in range(len(labels)):
            if labels[i] ==1:
                if result1[i][0]>0.5:
                    count1 += 1
            else:
                if result1[i][0]<=0.5:
                    count1 += 1

       
        # print("results",results)
        labels = labels.reshape(-1,anomaly_num)
        
        results = result1
        results = results.reshape(-1,anomaly_num)
        count2 = 0
             


        labels_index = labels.argmax(dim=1)
        results = results.argmax(dim=1)
        count2 = [0 for i in range(labels.shape[1])] # count is an array for every defect class
        for i,label in enumerate(labels_index):
            if results[i]==label:
                for j in range(len(count2)):
                    if label==j:
                        count2[j]+=1
        return count1/anomaly_num, count2, anomalys_support_new # return count1 for binary classification accuracy， count2 for classification accuracy
    # anomalys_support_new for a dict of index to anomaly
    
    def predict_for_metric(self, dataloader):
        label2index = {"bent": 0, "color": 1, "flip": 2, "scratch": 3}
        images, anomaly = [], []
        with tqdm.tqdm(dataloader, desc="Classification...", leave=False) as data_iterator:
            for i, item in enumerate(data_iterator):
                for j in range(item['image'].shape[0]):
                    if ("000" not in item['image_name'][j] and "001" not in item['image_name'][j]) and item['anomaly'][
                        j] != "good":
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly'][j]
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
            for k in range(len(anomaly)):
                anomaly[k] = label2index[anomaly[k]]
            images = torch.stack(images, dim=0)
            return self._predict_for_metric(images, anomaly)

    def _predict_for_metric(self, images, anomaly):
        """Infer score and mask for a batch of images."""
        self.eval_mode()
        query_features = self._embed(images.to(self.device))
        query_features = np.asarray(query_features)
        image_scores, query_distances, query_nns = self.anomaly_scorer.predict([query_features])
        query_nns = np.squeeze(query_nns, 1)
        nearest_neighbors_features = self.features[query_nns]
        query_residual_features = query_features - nearest_neighbors_features
        query_residual_features = query_residual_features.reshape(images.shape[0], -1)

        metric_model = self.metric_model.to(self.device)

        batch_size = self.residual_features_bank.shape[0]
        residual_features_bank = torch.from_numpy(self.residual_features_bank).reshape(-1, 28, 28, 1024).to(self.device).requires_grad_(True)
        residual_features_bank = residual_features_bank.permute(0, 3, 1, 2)
        
        # residual_features_bank = self.simple_proj(residual_features_bank)
        bank_embeddings = metric_model(residual_features_bank).reshape(batch_size, -1)
        # bank_embeddings = bank_embeddings / torch.norm(bank_embeddings, dim=-1, keepdim=True)
        
        batch_size_for_query= query_residual_features.shape[0]
        query_residual_features = torch.from_numpy(query_residual_features).reshape(-1, 28, 28, 1024).to(self.device).requires_grad_(True)
        query_residual_features = query_residual_features.permute(0, 3, 1, 2)
        # results_embeddings = self.simple_proj(query_residual_features)
        results_embeddings = metric_model(query_residual_features).reshape(batch_size_for_query, -1)
        # results_embeddings = results_embeddings / torch.norm(results_embeddings, dim=-1, keepdim=True)
        
        distances = self.featuresampler._compute_batchwise_differences(results_embeddings,
                                                                       bank_embeddings)
        labels = F.one_hot(torch.tensor(anomaly)).to(torch.float).to(self.device)
        count = 0
        for j in range(query_residual_features.shape[0]):
            if distances[j].argmin() == labels[j].argmax():
                count += 1
            print(count / query_residual_features.shape[0])
        return distances

    def predict_for_direct_classfication(self, dataloader,shot_num):
        #label2index = {"bent": 0, "color": 1, "flip": 2, "scratch": 3}
        with tqdm.tqdm(dataloader, desc="Classification by direct ConvNet...", leave=False) as data_iterator:
            images, anomaly = [], []
            for i, item in enumerate(data_iterator):
                for j in range(item['image'].shape[0]):
                    flag = False
                    for k in range(shot_num):
                        if SHOT_LIST[k] in item['image_name'][j]:
                            flag = True
                    if not flag and item['anomaly'][
                        j] != "good":
                        tmp_image = item['image'][j]
                        tmp_anomaly = item['anomaly'][j]
                        images.append(tmp_image)
                        anomaly.append(tmp_anomaly)
            # for k in range(len(anomaly)):
            #     anomaly[k] = label2index[anomaly[k]]
            for k in range(len(anomaly)):
                anomaly[k] = self.anomaly_enum_for_mlp[anomaly[k]]
            images = torch.stack(images, dim=0)
            count = 0
            if images.shape[0]>10:
                num = images.shape[0]//10
                yu = images.shape[0]%10
                for l in range(num):
                    result = self._predict_for_direct_classfication(images[l*10:(l+1)*10], anomaly[l*10:(l+1)*10])
                    count += result
                if yu!= 0:
                    result = self._predict_for_direct_classfication(images[num*10:], anomaly[num*10:])
                    count +=result
            else:
                result = self._predict_for_direct_classfication(images, anomaly)
                count +=result
            print("direct classification by ConvNet",count/images.shape[0])
            return 


    def _predict_for_direct_classfication(self, images, anomaly):
        """Infer score and mask for a batch of images."""
        results = self.backbone_for_direct_classification(images.to(self.device))
        labels = F.one_hot(torch.tensor(anomaly)).to(torch.float).to(self.device)
        count = 0
        for j in range(images.shape[0]):
            if results[j].argmax() == labels[j].argmax():
                count += 1
        return count

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            "features":self.features
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
            self,
            load_path: str,
            device: torch.device,
            nn_method: patchcore.common.FaissNN(False, 4),
            prepend: str = "",
            if_pretrained_net_for_backbone = True,
            
            
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"],if_pretrained_net_for_backbone
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        features = patchcore_params["features"]
        self.load_features(features)

        self.anomaly_scorer.load(load_path, prepend)
    def load_features(self,features):
        self.features = features

    def eval_mode(self):
        if isinstance(self.backbone, list):
            for i in range(len(self.backbone)):
                _ = self.forward_modules_list[i].eval()
        else:
            _ = self.forward_modules.eval()

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

