#%%
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from lxmert.lxmert.src.lxmert_lrp import LxmertAttention
from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import (GeneratorOurs, GeneratorBaselines,
                                                    GeneratorOursAblationNoAggregation)
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import visualization
import requests

import os
import glob
import sys
import pandas as pd

DEVICE = "cpu"


OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"



class ModelUsage:
    def __init__(self, use_lrp=False):
        self.vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = DEVICE

        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config = self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        if use_lrp:
            self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased").to(DEVICE)
        else:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to(DEVICE)

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        # self.vqa_dataset = vqa_data.VQADataset(splits="valid")

    def forward(self, item):
        URL, question = item

        self.image_file_path = URL

        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(URL)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            question,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        self.image_boxes_len = features.shape[1]
        self.bboxes = output_dict.get("boxes")
        self.output = self.lxmert_vqa(
            input_ids=inputs.input_ids.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            visual_feats=features.to(DEVICE),
            visual_pos=normalized_boxes.to(DEVICE),
            token_type_ids=inputs.token_type_ids.to(DEVICE),
            return_dict=True,
            output_attentions=False,
        )
        return self.output


def save_image_vis(model_lrp, image_file_path, bbox_scores):
    # bbox_scores = image_scores
    _, top_bboxes_indices = bbox_scores.topk(k=1, dim=-1)
    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    for index in range(len(bbox_scores)):
        [x, y, w, h] = model_lrp.bboxes[0][index]
        curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
        new_score_tensor = torch.ones_like(curr_score_tensor)*bbox_scores[index].item()
        mask[int(y):int(h), int(x):int(w)] = torch.max(new_score_tensor,mask[int(y):int(h), int(x):int(w)])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.unsqueeze_(-1)
    mask = mask.expand(img.shape)
    img = img * mask.cpu().data.numpy()
    cv2.imwrite('lxmert/lxmert/experiments/paper/new.jpg', img)


def test_save_image_vis(model_lrp, image_file_path, bbox_scores, evs, layer_num):
    # print(bbox_scores)
    # bbox_scores = image_scores
    _, top_bboxes_indices = bbox_scores.topk(k=5, dim=-1)

    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    for index in top_bboxes_indices:
        img = cv2.imread(image_file_path)
        [x, y, w, h] = model_lrp.bboxes[0][index]
        cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 10)
        cv2.imwrite('saved_images/{}.jpg'.format(index), img)

    count = 1
    plt.figure(figsize=(15, 10))

    for idx in top_bboxes_indices:
      idx = idx.item()
      plt.subplot(1, len(top_bboxes_indices), count)
      plt.title(str(idx) + " spectral " + evs + " " + layer_num)
      plt.axis('off')
      plt.imshow(cv2.imread('saved_images/{}.jpg'.format(idx)))
      count += 1

# def text_map(model_lrp, text_scores, layer_num):
#     plt.title("SA word impotance " + layer_num)
#     plt.xticks(np.arange(len(text_scores)), model_lrp.question_tokens[:])
#     plt.imshow(text_scores.unsqueeze(dim = 0).numpy())
#     plt.colorbar(orientation = "horizontal")
      
def text_map(model_lrp, text_scores):
    n_layers = len(text_scores)
    plt.figure(figsize=(15, 10))
    for j in range(len(text_scores)):
        # if j == 3:
            # print(text_scores[j])
        plt.subplot(3, n_layers - 3, j + 1)
        plt.title("SA word impotance " + str(j))
        plt.xticks(np.arange(len(text_scores[j])), model_lrp.question_tokens[1:-1])
        # plt.imshow(torch.abs(text_scores[j].unsqueeze(dim = 0)).numpy())
        plt.imshow(text_scores[j].unsqueeze(dim = 0).numpy())
        plt.colorbar(orientation = "horizontal")


        # plt.title("SA word impotance " + str(j))
        # plt.imshow(text_scores[j].unsqueeze(dim = 0).numpy())
        # plt.colorbar(orientation = "horizontal")

def viz_eigenvalues_dist (eigenvalues):
    print(len(eigenvalues[0]))
    n_layers = len(eigenvalues)
    # plt.figure(figsize=(15, 10))

    # fig, axes = plt.subplots(ncols=2, nrows=3)
    global_min = float('inf')
    global_max = float('-inf')

    for i in range(len(eigenvalues)):
        minval = torch.min(eigenvalues[i].real)
        maxval = torch.max(eigenvalues[i].real)
        global_min = min(global_min, minval)
        global_max = max(global_max, maxval)

    print(f"Global min: {global_min} and Global max: {global_max}")

    # for i, ax in zip(range(n_layers), axes.flat):
    # # for i in range(3):
    #     # for j in range(2):
    #     # print(eigenvalues[i].real)
    #     # minval = torch.min(eigenvalues[i].real)
    #     # maxval = torch.max(eigenvalues[i].real)

    #     # sns.kdeplot(x = eigenvalues[i].real.cpu().numpy(), y = np.arange(global_min, global_max, 0.01), ax = ax)
    #     # sns.kdeplot(x = eigenvalues[i].real.cpu().numpy(), ax = ax)
    #     df = pd.Series(eigenvalues[i].real, name = "EVals")
    #     sns.histplot(data = df, x = "EVals", kde = True)

    fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (8, 6))
    count_idx = 0
    for i in range(ax.shape[0]):
        for j in range(0, ax.shape[1]):
            # df = pd.DataFrame(eigenvalues[count_idx].real)
            df = pd.Series(eigenvalues[i].real, name = "EVals")

            sns.histplot(data = df, kde = True, ax = ax[i][j])
            count_idx += 1




    # plt.show()


    # for j in range(len(eigenvalues)):
    #     plt.subplot(3, n_layers - 3, j + 1)
    #     # plt.title("Eigenvalues Distribution in Layer - " + str(j))
    #     sns.displot(eigenvalues[j].real.cpu().numpy(), kind = "kde")



def spectral_stuff():
    model_lrp = ModelUsage(use_lrp=True)
    lrp = GeneratorOurs(model_lrp)
    # baselines = GeneratorBaselines(model_lrp)
    vqa_answers = utils.get_data(VQA_URL)

    # baselines.generate_transformer_attr(None)
    # baselines.generate_attn_gradcam(None)
    # baselines.generate_partial_lrp(None)
    # baselines.generate_raw_attn(None)
    # baselines.generate_rollout(None)

    image_ids = [
        # giraffe
        'COCO_val2014_000000185590',
        # baseball
        'COCO_val2014_000000127510',
        # bath
        'COCO_val2014_000000324266',
        # frisbee
        'COCO_val2014_000000200717',

        'COCO_val2014_000000159282',

        'COCO_val2014_000000134886',

        'COCO_val2014_000000456784', 

        'COCO_val2014_000000085101',

        'COCO_val2014_000000254834',

        'COCO_val2014_000000297681',

        'COCO_val2014_000000193112',

        'COCO_val2014_000000312081',

        'COCO_val2014_000000472530',

        'COCO_val2014_000000532164',

        'COCO_val2014_000000009466',

        'COCO_val2014_000000435187',

        'COCO_val2014_000000353405',

        'COCO_val2014_000000516414',

        'COCO_val2014_000000097693',

        'COCO_val2014_000000014450',

        'COCO_val2014_000000008045', ##custom

        'COCO_val2014_000000016499', ##custom,

        'COCO_val2014_000000297180',

        "D:\Thesis_2023-24\weird_tejju.jpg"
    ]

    test_questions_for_images = [
        ################## paper samples
        # giraffe
        "is the animal eating?",
        # baseball
        "did he catch the ball?",
        # bath
        "is the tub white ?",
        # frisbee
        "did the man just catch the frisbee?",

        # "What kind of flowers are those?"
        "What is at the bottom of the vase?",

        "How many planes are in the air?",

        "What kind of cake is that?",

        "Are there clouds in the picture?",

        "What is reflecting in the building's windows?",

        "Why are the lights reflecting?",

        "What is the person riding?", # failure
 
        "How many kids have their hands up in the air?", # both weird
        ################## paper samples

        "Is there a microwave in the room?",

        "Which of the people is wearing a hat that would be appropriate for St. Patrick's Day?",

        "How many shoes do you see?",

        "What surrounds the vehicle?",

        "How many clocks?",

        "Are these yachts?",

        "What color are blankets on this bed?",

        "Is this a railroad track?",

        "Where is the sink and where is the bathtub?",

        "Is there a train?",

        "Where are they?",

        "Is there a jacket?"
    ]

    # URL = 'lxmert/lxmert/experiments/paper/{0}/{0}.jpg'.format(image_ids[4])
    URL = '../../data/root/val2014/{}.jpg'.format(image_ids[0])
    # URL = image_ids[-1]
    # URL = 'giraffe.jpg'
    qs = test_questions_for_images[0]
    R_t_t, R_t_i, ei, et = lrp.generate_ours_dsm((URL, qs), sign_method="mean", how_many = 10, use_lrp=False, 
              
                                         normalize_self_attention=True, method_name="dsm")
    # text_scores = R_t_t
    # image_scores = R_t_i

    # print(f"Shape of text scores: {len(text_scores)}")
    # print(eigenvalues)
    viz_eigenvalues_dist(ei)
    viz_eigenvalues_dist(et)

    # for i in range(len(image_scores)):    

        # text_map(model_lrp, text_scores)
        # test_save_image_vis(model_lrp, URL, image_scores[i], "+", str(i))
    # test_save_image_vis(model_lrp, URL, image_scores * -1, "-")
        
    # for j in range(len(text_scores)):
    # text_map(model_lrp, text_scores)



    # save_image_vis(model_lrp, URL, image_scores)
    # orig_image = Image.open(model_lrp.image_file_path)
    # plt.imshow(text_scores.unsqueeze(dim = 0).numpy())

    # fig, axs = plt.subplots(ncols=3, figsize=(20, 5))
    # axs[0].imshow(orig_image)
    # axs[0].axis('off')
    # axs[0].set_title('original')

    # masked_image = Image.open('lxmert/lxmert/experiments/paper/new.jpg')
    # axs[1].imshow(masked_image)
    # axs[1].axis('off')
    # axs[1].set_title('masked')

    # axs[2].imshow(R_t_i.unsqueeze(dim = 0).numpy())
    # axs[2].set_xlabel("object number")
    # axs[2].set_title('object relevance')

    # text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
    # vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,model_lrp.question_tokens[1:-1],1)]
    # visualization.visualize_text(vis_data_records)
    print(f"QUESTION: {qs}")
    print("ANSWER:", vqa_answers[model_lrp.output.question_answering_score.argmax()])
    

    plt.show()


if __name__ == '__main__':
    # main()


    files = glob.glob('saved_images/*')
    for f in files:
        os.remove(f)
    spectral_stuff()