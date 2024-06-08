from IPython.display import Image, display
import PIL.Image
import io
import torch
import numpy as np
from VisualBERT.processing_image import Preprocess
# from visualizing_image import SingleImageViz
from VisualBERT.modeling_frcnn import GeneralizedRCNN
from VisualBERT.utils import Config
import VisualBERT.utils as utils
from transformers import BertTokenizerFast

from VisualBERT.mmf.models.visual_bert import VisualBERT

# from ExplanationGenerator import GeneratorOurs
import cv2
import matplotlib.pyplot as plt
import os
import glob

import yaml
import json
# from dotdict import DotDict
from dotmap import DotMap
from yaml2object import YAMLObject
from omegaconf import OmegaConf


# URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"
# URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

DEVICE = "cpu"


# "D:\Programs\anaconda3\Lib\site-packages\transformers\models\visual_bert\modeling_visual_bert.py"


class ModelUsage:
    def __init__(self, config):
        self.config = config
        # self.vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        # self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")


        # load models and model components
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = DEVICE


        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        # self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config = self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)

        # self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # with open('classification/yaml', 'r') as f:
            # config = yaml.load(f, Loader=yaml.SafeLoader)



        # self.visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa", output_hidden_states=True).to(DEVICE)
        self.visualbert_vqa = VisualBERT(self.config)



        self.visualbert_vqa.eval()

        self.model = self.visualbert_vqa

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


        inputs = self.bert_tokenizer(
            question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # print(inputs)

        features = output_dict.get("roi_features")
        # self.image_boxes_len = features.shape[1]
        self.bboxes = output_dict.get("boxes")
        # normalized_boxes = output_dict.get("normalized_boxes")

        sample_list = {}
        sample_list['input_ids'] = inputs.input_ids.to(DEVICE)
        sample_list['input_mask'] = inputs.attention_mask.to(DEVICE)
        # sample_list['token_type_ids'] = inputs.token_type_ids.to(DEVICE)
        sample_list['segment_ids'] = inputs.token_type_ids.to(DEVICE)

        sample_list['image_feature_0'] = features.to(DEVICE)
        # sample_list['input_ids'] = inputs.input_ids


        # self.output_vqa = self.visualbert_vqa(
        #     input_ids=inputs.input_ids.to(DEVICE),
        #     attention_mask=inputs.attention_mask.to(DEVICE),
        #     visual_embeds=features.to(DEVICE),
        #     visual_attention_mask=torch.ones(features.shape[:-1]).to(DEVICE),
        #     token_type_ids=inputs.token_type_ids.to(DEVICE),
        #     output_attentions=False
        # )

        self.output_vqa = self.visualbert_vqa(sample_list)

        # inputs = self.lxmert_tokenizer(
        #     question,
        #     truncation=True,
        #     return_token_type_ids=True,
        #     return_attention_mask=True,
        #     add_special_tokens=True,
        #     return_tensors="pt"
        # )
        # self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        # self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized
        # features = output_dict.get("roi_features")
        # self.image_boxes_len = features.shape[1]
        # self.bboxes = output_dict.get("boxes")
        # self.output = self.lxmert_vqa(
        #     input_ids=inputs.input_ids.to(DEVICE),
        #     attention_mask=inputs.attention_mask.to(DEVICE),
        #     visual_feats=features.to(DEVICE),
        #     visual_pos=normalized_boxes.to(DEVICE),
        #     token_type_ids=inputs.token_type_ids.to(DEVICE),
        #     return_dict=True,
        #     output_attentions=False,
        # )

        return self.output_vqa
    

def test_save_image_vis(model_lrp, image_file_path, bbox_scores, evs):
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
      plt.title(str(idx) + " " + evs)
      plt.axis('off')
      plt.imshow(cv2.imread('saved_images/{}.jpg'.format(idx)))
      count += 1


# class Config( DotDict ):
#     def __init__( self ):
#         with open('VisualBERT/mmf/configs/models/visual_bert/classification.yaml') as file:
#             DotDict.__init__(self, yaml.safe_load(file))


# class Config(metaclass=YAMLObject):
#     source = 'VisualBERT/mmf/configs/models/visual_bert/classification.yaml'

# class Config:
#     def __init__(self, path):
#         self.path = path
#     def get_config_dict(self):
#         with open(self.path, 'r') as f:
#             config = yaml.load(f, Loader=yaml.SafeLoader)
#         return config
#     def get_config_object(self):
#         config = self.get_config_dict()



# config = Config()
# print(config.django.admin.user)

def spectral_method ():
    # with open('VisualBERT/mmf/configs/models/visual_bert/classification.yaml', 'r') as f:
    #     config = yaml.load(f, Loader=yaml.SafeLoader)
    # m = DotMap(config)

    conf = OmegaConf.load('VisualBERT/mmf/configs/models/visual_bert/classification.yaml')
    print(conf.model_config)
    
    # config = Config()
    # print(Config.to_dict().model_config.visual_bert)
    # with open('VisualBERT/mmf/configs/models/visual_bert/classification.yaml') as f:
    #     config = json.dumps(yaml.load(f, Loader=yaml.SafeLoader))
    # print(m.model_config.visual_bert.training_head_type)
    model = ModelUsage(conf.model_config.visual_bert)
    # ours = GeneratorOurs(model)
    vqa_answers = utils.get_data(VQA_URL)


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






    URL = '../../data/root/val2014/{}.jpg'.format(image_ids[0])
    item = (URL, test_questions_for_images[0])
    output = model.forward(item)
    print(f"Len: {output['scores'][0]}")
    print(vqa_answers[output['scores'][0].argmax()])
    # print(vqa_answers[output.logits.argmax()])

    # R_t_t, R_t_i = ours.generate_ours(item)
    # image_scores = R_t_i
    # text_scores = R_t_t
    # print(f"Size of R_t_i: {image_scores.size()} | Size of R_t_t: {text_scores.size()}")
    # test_save_image_vis(model, URL, image_scores, "HC RM")

    # plt.show()

    # print(R)


if __name__ == '__main__':
    # main()
    files = glob.glob('saved_images/*')
    for f in files:
        os.remove(f)
    spectral_method()