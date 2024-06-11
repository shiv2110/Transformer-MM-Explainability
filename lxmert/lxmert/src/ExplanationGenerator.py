import numpy as np
import torch
import copy
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig

import torch.nn.functional as F
from torch.nn import Linear

from pymatting.util.util import row_sum
from scipy.sparse import diags
from scipy.stats import skew
from .eigenshuffle import eigenshuffle
import math
# from sentence_transformers import SentenceTransformer
# from torch.nn import CosineSimilarity as CosSim

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def avg_heads_new(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = cam.clamp(min=0).mean(dim=0)
    grad = grad.clamp(min=0).mean(dim=0)

    cam = grad @ cam.T
    # cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rules 10 + 11 from paper
def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_ss_addition = torch.matmul(cam_sq, R_qs)
    return R_sq_addition, R_ss_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    # computing R hat
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    # normalizing R hat
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention



class GeneratorOurs:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization
        # self.image_R = []
        # self.text_R = []
        # self.text_image_R = []
        # self.image_text_R = []
        # self.self_attn_lang_grads = []
        # self.self_attn_image_grads = []
        # self.co_self_attn_lang_grads = []
        # self.co_self_attn_image_grads = []

        # self.self_attn_lang_agg = []
        # self.self_attn_image_agg = []
        # self.co_attn_lang_agg = []
        # self.co_attn_image_agg = []

        # # self.attn_t_i = []
        # self.attn_i_t = []
        # self.attn_t_t = []
        # self.attn_grads_t_i = []
        # self.attn_grads_i_t = []

        # self.all_attn_t_i = []

        # self.cross_attn_viz_feat = []
        # self.cross_attn_lg_feat = []

        # self.attn_viz_feats = []
        # self.cross_attn_viz_feat_list = []

        # self.lrp_R_t_i = []



    def handle_self_attention_lang(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            
            R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
            self.R_t_t += R_t_t_add
            self.R_t_i += R_t_i_add
            # self.text_image_R.append(self.R_t_i.detach().clone())
            # self.text_R.append(self.R_t_t.detach().clone())
            # self.self_attn_lang_grads.append(grad)
            # self.self_attn_lang_agg.append(cam)
        # print(f"length of text_R: {len(self.text_R)}")


    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
            self.R_i_i += R_i_i_add
            self.R_i_t += R_i_t_add
            # self.image_text_R.append(self.R_i_t.detach().clone())
            # self.image_R.append(self.R_i_i.detach().clone())
            # self.self_attn_image_grads.append(grad)
            # self.self_attn_image_agg.append(cam)



    def handle_co_attn_self_lang(self, block):
        grad = block.lang_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.lang_self_att.self.get_attn_cam().detach()
        else:
            cam = block.lang_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add
        # print(self.R_t_t)
        # self.text_R.append(self.R_t_t.detach().clone())
        # self.text_image_R.append(self.R_t_i.detach().clone())
        # self.co_self_attn_lang_grads.append(grad)
        # self.co_attn_lang_agg.append(cam)

    # def handle_co_attn_self_lang(self, block):
    #         grad = block.lang_self_att.self.get_attn_gradients().detach()
    #         # if self.use_lrp:
    #         cam = block.lang_self_att.self.get_attn_cam().detach()
    #         self.lrp_R_t_i.append(cam)
    #         # else:
    #         cam = block.lang_self_att.self.get_attn().detach()
    #         cam = avg_heads(cam, grad)
    #         R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
    #         self.R_t_t += R_t_t_add
    #         self.R_t_i += R_t_i_add
    #         # print(self.R_t_t)
    #         self.text_R.append(self.R_t_t.detach().clone())
    #         self.text_image_R.append(self.R_t_i.detach().clone())
    #         self.co_self_attn_lang_grads.append(grad)
    #         self.co_attn_lang_agg.append(cam)



    def handle_co_attn_self_image(self, block):
        grad = block.visn_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.visn_self_att.self.get_attn_cam().detach()
        else:
            cam = block.visn_self_att.self.get_attn().detach()
            # print(f"cam shape: {cam.shape}")
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add
        # self.image_R.append(self.R_i_i.detach().clone())
        # self.image_text_R.append(self.R_i_t.detach().clone())
        # self.co_self_attn_image_grads.append(grad)
        # self.co_attn_image_agg.append(cam)
    

    

    def handle_co_attn_lang(self, block):
        if self.use_lrp:
            cam_t_i = block.visual_attention.att.get_attn_cam().detach()
        else:
            cam_t_i = block.visual_attention.att.get_attn().detach()
        
        # self.all_attn_t_i.append(cam_t_i)
        grad_t_i = block.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        # self.attn_t_i.append(cam_t_i)
        # self.attn_grads_t_i.append(grad_t_i)
        return R_t_i_addition, R_t_t_addition

    def handle_co_attn_image(self, block):
        if self.use_lrp:
            cam_i_t = block.visual_attention_copy.att.get_attn_cam().detach()
        else:
            cam_i_t = block.visual_attention_copy.att.get_attn().detach()
        grad_i_t = block.visual_attention_copy.att.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        # self.attn_i_t.append(cam_i_t)
        # self.attn_grads_i_t.append(grad_i_t)
        return R_i_t_addition, R_i_i_addition
    


    def generate_ours(self, input, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True, method_name="ours"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # self.attn_t_i = []
        # self.text_R = []


        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # print(f"HEloooo {output}, {output.size()}")
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot * output) #baka
        one_hot = torch.sum(one_hot.cuda() * output) #baka

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        if self.use_lrp:
            model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        self.handle_self_attention_lang(blocks)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        self.handle_self_attention_image(blocks)
        # self.attn_viz_feats = model.lxmert.encoder.visual_feats_list_r

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        # self.cross_attn_viz_feat_list = model.lxmert.encoder.visual_feats_list_x

        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(blk)

            self.R_t_i += R_t_i_addition
            self.R_t_t += R_t_t_addition
            self.R_i_t += R_i_t_addition
            self.R_i_i += R_i_i_addition

            # self.text_image_R.append(self.R_t_i)
            # self.text_R.append(self.R_t_t)
            # self.image_text_R.append(self.R_i_t)
            # self.image_R.append(self.R_i_i)

            # language self attention
            self.handle_co_attn_self_lang(blk)

            # image self attention
            self.handle_co_attn_self_image(blk)

            # self.cross_attn_viz_feat.append(blk.cross_attn_visual_feats.detach().clone())
            # self.cross_attn_lg_feat.append(blk.cross_attn_lang_feats.detach().clone())

            # self.cross_attn_viz_feat.append(blk.visual_attention_copy.att.cross_attn_visual_feats)

        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]

        # self.cross_attn_viz_feat.append(blk.cross_attn_visual_feats.detach().clone())
        # self.cross_attn_lg_feat.append(blk.cross_attn_lang_feats.detach().clone())
        # cross attn- first for language then for image
        R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
        self.R_t_i += R_t_i_addition
        self.R_t_t += R_t_t_addition

        # self.text_image_R.append(self.R_t_i.detach().clone())
        # self.text_R.append(self.R_t_t.detach().clone())

        # language self attention
        self.handle_co_attn_self_lang(blk)

        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        # self.R_t_i[0, 0] = 0
        # self.text_R[-1][0, 0] = 0
        # print(self.R_t_i[0][0])
        return self.R_t_t, self.R_t_i #baka
        # return self.R_t_i.T, self.R_i_t.T #baka


    def handle_fev (self, fev):
        temp = torch.abs(fev)
        idx = torch.argmax(temp)
        if fev[idx] < 0:
            return fev * -1
        return fev


    def generate_ours_dsm(self, input, how_many = 5, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True, 
                          method_name="dsm"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10

        output = self.model_usage.forward(input).question_answering_score
        # print(f"{output.last_hidden_state.shape}")
        model = self.model_usage.model

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output) #baka
        # one_hot = torch.sum(one_hot.cuda() * output) #baka

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        # blocks = model.lxmert.encoder.x_layers
        
        # image_feats = model.lxmert.encoder.visual_feats_list_x[-2].detach().clone()
        # image_flen = len(model.lxmert.encoder.visual_feats_list_x)
        # text_flen = len(model.lxmert.encoder.lang_feats_list_x)
        # text_feats = model.lxmert.encoder.lang_feats_list_x[-2].detach().clone()

        # blk_count = 0
        def get_eigs (feats_list, modality, how_many):

            if modality == "image":
                feats = F.normalize(feats_list.detach().clone().squeeze().cpu(), p = 2, dim = -1)
            else:
                feats = F.normalize(feats_list.detach().clone().squeeze().cpu(), p = 2, dim = -1)[1:-1]
                # feats1 = feats

            W_feat = (feats @ feats.T)
            W_feat = (W_feat * (W_feat > 0))
            W_feat = W_feat / W_feat.max() 

            W_feat = W_feat.cpu().numpy()

            def get_diagonal (W):
                D = row_sum(W)
                D[D < 1e-12] = 1.0  # Prevent division by zero.
                D = diags(D)
                return D
            
            D = np.array(get_diagonal(W_feat).todense())

            L = D - W_feat
            # L[L < 0] = 0
            L_shape = L.shape[0]
            if how_many >= L_shape - 1:
                how_many = L_shape - 2

            try:
                eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5, M = D)
            except:
                try:
                    eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5)
                except:
                    eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM')
            eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
            
            n_tuple = torch.kthvalue(eigenvalues.real, 2)
            fev_idx = n_tuple.indices
            fev = eigenvectors[fev_idx]


            # return torch.abs(fev)
            # return fev
            fev = self.handle_fev(fev)
            if modality == 'text':
                fev = torch.abs(fev)
                fev = torch.cat( ( torch.zeros(1), fev, torch.zeros(1)  ) )
                # fev[0], fev[] = -1, -1
            return fev


        
        image_fev = get_eigs(model.lxmert.encoder.visual_feats_list_x[-2], 
                                                 "image", how_many)
        
        lang_fev = get_eigs(model.lxmert.encoder.lang_feats_list_x[-1], 
                                               "text", how_many)

        # return lang_fevs[-2], image_fevs[-2], eigenvalues_image, eigenvalues_text
        return lang_fev, image_fev



    def get_diagonal (self, W):
        D = row_sum(W)
        D[D < 1e-12] = 1.0  # Prevent division by zero.
        D = diags(D)
        return D


    def get_fev (self, feats, modality, how_many = None):
        if feats.size(0) == 1:
            feats = feats.detach().squeeze()


        if modality == "image":
            n_image_feats = feats.size(0)
            val = int( math.sqrt(n_image_feats) )
            if val * val == n_image_feats:
                feats = F.normalize(feats, p = 2, dim = -1)
            elif val * val + 1 == n_image_feats:
                feats = F.normalize(feats, p = 2, dim = -1)[1:]
            else:
                print(f"Invalid number of features detected: {n_image_feats}")

        else:
            feats = F.normalize(feats, p = 2, dim = -1)[1:-1]

        W_feat = (feats @ feats.T)
        W_feat = (W_feat * (W_feat > 0))
        W_feat = W_feat / W_feat.max() 

        W_feat = W_feat.detach().cpu().numpy()

        
        D = np.array(self.get_diagonal(W_feat).todense())

        L = D - W_feat

        L_shape = L.shape[0]
        if how_many >= L_shape - 1: #add this in meter
            how_many = L_shape - 2

        try:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5, M = D)
        except:
            try:
                eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5)
            except:
                eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM')
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
        
        n_tuple = torch.kthvalue(eigenvalues.real, 2)
        fev_idx = n_tuple.indices
        fev = eigenvectors[fev_idx]
        # fev = eigenvectors[1]

        # return torch.abs(fev)
        # return fev 
        fev = self.handle_fev(fev)
        if modality == 'text':
            fev = torch.abs(fev)
            fev = torch.cat( ( torch.zeros(1), fev, torch.zeros(1) ) )
            # fev[0], fev[-1] = -1, -1
        return fev




    def generate_ours_dsm_grad(self, input, how_many = 5, index=None):
        
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot * output) #baka
        one_hot = torch.sum(one_hot.cuda() * output) #baka
        model.zero_grad()
        one_hot.backward(retain_graph=True)


        image_flen1 = len(model.lxmert.encoder.visual_feats_list_x)
        text_flen1 = len(model.lxmert.encoder.lang_feats_list_x)

        def get_eigs (feats_list, flen, modality, how_many):
            # blk_count = 0
            layer_wise_fevs = []
            blk = model.lxmert.encoder.x_layers
            # lang_blk = model.lxmert.encoder.layer

            for i in range(flen):
                # feats = F.normalize(feats_list[i].detach().clone().squeeze().cpu(), p = 2, dim = -1)
                # print(f"Features' shape: {feats.shape}")
                fev = self.get_fev(feats_list[i], modality, how_many)
                if modality == "text":
                    fev = fev[1:-1]
                    # fev = torch.cat( ( torch.zeros(1), fev ) )

                # layer_wise_fevs.append( eigenvalues[fev_idx].real * fev )
                if modality == "image":
                    grad = blk[i].visn_self_att.self.get_attn_gradients().detach()
                    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                    grad = grad.clamp(min=0).mean(dim=0)
                    fev = fev.to(model.device)
                    fev = grad @ fev.unsqueeze(1)
                    fev = fev[:, 0]

                else:
                    grad = blk[i].lang_self_att.self.get_attn_gradients().detach()[:, :, 1:-1, 1:-1]
                    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                    grad = grad.clamp(min=0).mean(dim=0)
                    fev = fev.to(model.device)
                    fev = grad @ fev.unsqueeze(1)
                    fev = fev[:, 0]
                    fev = torch.cat( ( torch.zeros(1).to(model.device), fev, torch.zeros(1).to(model.device) ) )
                    # fev[0], fev[-1] = -1, -1


                # layer_wise_fevs.append( torch.abs(fev) )
                layer_wise_fevs.append( self.handle_fev(fev) )

                # layer_wise_fevs.append( self.handle_fev(fev) if modality == "image" else torch.abs(fev) )

      
            return layer_wise_fevs

        image_fevs = get_eigs(model.lxmert.encoder.visual_feats_list_x, 
                                                 image_flen1 - 1, "image", how_many)
        
        lang_fevs = get_eigs(model.lxmert.encoder.lang_feats_list_x, 
                                               text_flen1, "text", how_many)


        # return lang_fevs[-2], image_fevs[-2], eigenvalues_image, eigenvalues_text
        new_fev_image = torch.stack(image_fevs, dim=0).sum(dim=0)
        new_fev_lang = torch.stack(lang_fevs, dim=0).sum(dim=0)
        # new_fev1 = (new_fev1 - torch.min(new_fev1))/(torch.max(new_fev1) - torch.min(new_fev1))
        # new_fev = (new_fev - torch.min(new_fev))/(torch.max(new_fev) - torch.min(new_fev))

        return new_fev_lang, new_fev_image




    
    def generate_ours_dsm_grad_cam(self, input, how_many = 5, index=None):
        
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot * output) #baka
        # one_hot = torch.sum(one_hot.cuda() * output) #baka
        model.zero_grad()
        one_hot.backward(retain_graph=True)


        image_flen1 = len(model.lxmert.encoder.visual_feats_list_x)
        text_flen1 = len(model.lxmert.encoder.lang_feats_list_x)

        def get_eigs (feats_list, flen, modality, how_many):
            # blk_count = 0
            layer_wise_fevs = []
            blk = model.lxmert.encoder.x_layers
            # lang_blk = model.lxmert.encoder.layer

            for i in range(flen):
                # feats = F.normalize(feats_list[i].detach().clone().squeeze().cpu(), p = 2, dim = -1)
                # print(f"Features' shape: {feats.shape}")
                fev = self.get_fev(feats_list[i], modality, how_many)
                if modality == "text":
                    fev = fev[1:-1]
                    # fev = torch.cat( ( torch.zeros(1), fev ) )

                # layer_wise_fevs.append( eigenvalues[fev_idx].real * fev )
                if modality == "image":
                    grad = blk[i].visn_self_att.self.get_attn_gradients().detach()
                    # print()
                    cam = blk[i].visn_self_att.self.get_attn().detach()
                    cam = avg_heads(cam, grad)

                    # cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
                    # cam = cam.clamp(min=0).mean(dim=0)

                    # grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                    # grad = grad.clamp(min=0).mean(dim=0)
                    # print(f"GRAD SHAPE: {grad.size()}")

                    fev = fev.to(model.device)
                    # cam = grad @ cam
                    fev = cam @ fev.unsqueeze(1)
                    fev = fev[:, 0]

                else:
                    grad = blk[i].lang_self_att.self.get_attn_gradients().detach()[:, :, 1:-1, 1:-1]
                    cam = blk[i].lang_self_att.self.get_attn().detach()[:, :, 1:-1, 1:-1]
                    cam = avg_heads(cam, grad)
                    # cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
                    # cam = cam.clamp(min=0).mean(dim=0)

                    # grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                    # grad = grad.clamp(min=0).mean(dim=0)
                    # print(f"GRAD SHAPE: {grad.size()}")
                    fev = fev.to(model.device)
                    fev = cam @ fev.unsqueeze(1)
                    fev = fev[:, 0]
                    fev = torch.cat( ( torch.zeros(1).to(model.device), fev, torch.zeros(1).to(model.device)  ) )


                layer_wise_fevs.append( torch.abs(fev) )
      
            return layer_wise_fevs


        image_fevs = get_eigs(model.lxmert.encoder.visual_feats_list_x, 
                                                 image_flen1 - 1, "image", how_many)
        
        lang_fevs = get_eigs(model.lxmert.encoder.lang_feats_list_x, 
                                               text_flen1, "text", how_many)


        # return lang_fevs[-2], image_fevs[-2], eigenvalues_image, eigenvalues_text
        new_fev_image = torch.stack(image_fevs, dim=0).sum(dim=0)
        new_fev_lang = torch.stack(lang_fevs, dim=0).sum(dim=0)
        # new_fev1 = (new_fev1 - torch.min(new_fev1))/(torch.max(new_fev1) - torch.min(new_fev1))
        # new_fev = (new_fev - torch.min(new_fev))/(torch.max(new_fev) - torch.min(new_fev))

        return new_fev_lang, new_fev_image




    def generate_eigen_cam(self, input, how_many = 5, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True, 
                          method_name="dsm"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10

        # self.cross_attn_viz_feat = []
        # self.cross_attn_lg_feat = []
        # self.attn_viz_feats = []
        # self.cross_attn_viz_feat_list = []
        # self.cross_attn_lg_feat_list = []


        # print(len(self.cross_attn_viz_feat))
        # kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        # print(f"{output.last_hidden_state.shape}")
        model = self.model_usage.model

        model.zero_grad()
        
        # image_feats = model.lxmert.encoder.visual_feats_list_x[-2].detach().clone()
        # image_flen = len(model.lxmert.encoder.visual_feats_list_x)
        # text_flen = len(model.lxmert.encoder.lang_feats_list_x)
        # text_feats = model.lxmert.encoder.lang_feats_list_x[-2].detach().clone()


        # feats = F.normalize(feats, p = 2, dim = -1)
        # image_feats = image_feats.squeeze().cpu()
        # text_feats = text_feats.squeeze().cpu()[1:-1]
        

        def get_eigs (feats_list, modality, how_many):
            # layer_wise_fevs = []
            # layer_wise_eigenvalues = []
            # for i in range(flen):
                # feats = F.normalize(feats_list[i].detach().clone().squeeze().cpu(), p = 2, dim = -1)
                # print(f"Features' shape: {feats.shape}")
            if modality == "image":
                feats = F.normalize(feats_list.detach().clone().squeeze(), p = 2, dim = -1)

            else:
                feats = F.normalize(feats_list.detach().clone().squeeze(), p = 2, dim = -1)[1:-1]
                # feats1 = feats


            U, S, V = torch.linalg.svd(feats, full_matrices=False)
                # print(f"Right singular value: {V.shape}")
                # print(f"Left singular value: {U.shape}")
                # print(f"Diagonal matrix: {S.shape}")

                # eigenvectors = U[0][:, :how_many].T.to('cpu', non_blocking=True) # this is U matrix
                # eigenvalues = S[1][:how_many].to('cpu', non_blocking=True)
                # print(f"SVD EVs shape: {eigenvectors.shape}")
                # print(f"Eigenvalues type: {type(eigenvalues)}")
                # fev = eigenvectors[1]

                ########## eigenCAM style - baka #########################

                # fev = fev.unsqueeze(dim = 1)
                # fev = torch.abs(fev)
                # fev = fev.repeat(1, 768)
                # # print(f"fev unsueezed and repeated: {fev1.repeat(1, 768).shape}")
            cam = feats @ V.T
            # proj = Linear(cam.shape[0], 1)
            fev = cam[0]
                # print(S)
                # fev = fev.squeeze()
                # # print(f"net fev shape: {fev.shape}")


                ########## eigenCAM style - baka #########################


            if modality == 'text':
                fev = torch.cat( ( torch.zeros(1).to(model.device), fev, torch.zeros(1).to(model.device)  ) )
                    # fev = torch.cat( ( torch.zeros(1), fev ) )


            return torch.abs(fev)

        
        image_fev = get_eigs(model.lxmert.encoder.visual_feats_list_x[-2], 
                                                #  model.lxmert.encoder.lang_feats_list_x,
                                                 "image", how_many)
        
        lang_fev = get_eigs(model.lxmert.encoder.lang_feats_list_x[-2], 
                                               "text", how_many)

        return lang_fev, image_fev
        # return lang_fevs, image_fevs



class GeneratorOursAblationNoAggregation:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def handle_self_attention_lang(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
            self.R_t_t = R_t_t_add
            self.R_t_i = R_t_i_add

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
            self.R_i_i = R_i_i_add
            self.R_i_t = R_i_t_add

    def handle_co_attn_self_lang(self, block):
        grad = block.lang_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.lang_self_att.self.get_attn_cam().detach()
        else:
            cam = block.lang_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t = R_t_t_add
        self.R_t_i = R_t_i_add

    def handle_co_attn_self_image(self, block):
        grad = block.visn_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.visn_self_att.self.get_attn_cam().detach()
        else:
            cam = block.visn_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i = R_i_i_add
        self.R_i_t = R_i_t_add

    def handle_co_attn_lang(self, block):
        if self.use_lrp:
            cam_t_i = block.visual_attention.att.get_attn_cam().detach()
        else:
            cam_t_i = block.visual_attention.att.get_attn().detach()
        grad_t_i = block.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i, apply_normalization=self.normalize_self_attention)
        return R_t_i_addition, R_t_t_addition

    def handle_co_attn_image(self, block):
        if self.use_lrp:
            cam_i_t = block.visual_attention_copy.att.get_attn_cam().detach()
        else:
            cam_i_t = block.visual_attention_copy.att.get_attn().detach()
        grad_i_t = block.visual_attention_copy.att.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t, apply_normalization=self.normalize_self_attention)
        return R_i_t_addition, R_i_i_addition

    def generate_ours_no_agg(self, input, index=None, use_lrp=False, normalize_self_attention=True, method_name="ours_no_agg"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        if self.use_lrp:
            model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        self.handle_self_attention_lang(blocks)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        self.handle_self_attention_image(blocks)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(blk)

            self.R_t_i = R_t_i_addition
            self.R_t_t = R_t_t_addition
            self.R_i_t = R_i_t_addition
            self.R_i_i = R_i_i_addition

            # language self attention
            self.handle_co_attn_self_lang(blk)

            # image self attention
            self.handle_co_attn_self_image(blk)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn- first for language then for image
        R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
        self.R_t_i = R_t_i_addition
        self.R_t_t = R_t_t_addition

        # language self attention
        self.handle_co_attn_self_lang(blk)

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i


class GeneratorBaselines:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def generate_transformer_attr(self, input, index=None, method_name="transformer_attr"):
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot.cuda() * output)
        one_hot = torch.sum(one_hot * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)


        # image self attention
        blocks = model.lxmert.encoder.r_layers
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break

            # language self attention
            grad = blk.lang_self_att.self.get_attn_gradients().detach()
            cam = blk.lang_self_att.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)

            # image self attention
            grad = blk.visn_self_att.self.get_attn_gradients().detach()
            cam = blk.visn_self_att.self.get_attn_cam().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn_cam().detach()
        grad_t_i = blk.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        # self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        self.R_t_i = cam_t_i

        # language self attention
        grad = blk.lang_self_att.self.get_attn_gradients().detach()
        cam = blk.lang_self_att.self.get_attn_cam().detach()
        cam = avg_heads(cam, grad)
        self.R_t_t += torch.matmul(cam, self.R_t_t)

        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i

    def generate_partial_lrp(self, input, index=None, method_name="partial_lrp"):
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.zeros(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.zeros(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # last cross attention + self- attention layer
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn_cam().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_i = cam_t_i

        # language self attention
        cam = blk.lang_self_att.self.get_attn_cam().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        self.R_t_t = cam

        # normalize to get non-negative cams
        self.R_t_t = (self.R_t_t - self.R_t_t.min()) / (self.R_t_t.max() - self.R_t_t.min())
        self.R_t_i = (self.R_t_i - self.R_t_i.min()) / (self.R_t_i.max() - self.R_t_i.min())
        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def generate_raw_attn(self, input, method_name="raw_attention"):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.zeros(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.zeros(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        # last cross attention + self- attention layer
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        # self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        self.R_t_i = cam_t_i

        # language self attention
        cam = blk.lang_self_att.self.get_attn().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        self.R_t_t = cam

        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, input, index=None, method_name="gradcam"):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        # last cross attention + self- attention layer
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        grad_t_i = blk.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = blk.visual_attention.att.get_attn().detach()
        cam_t_i = self.gradcam(cam_t_i, grad_t_i)
        # self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        self.R_t_i = cam_t_i

        # language self attention
        grad = blk.lang_self_att.self.get_attn_gradients().detach()
        cam = blk.lang_self_att.self.get_attn().detach()
        self.R_t_t = self.gradcam(cam, grad)

        # disregard the [CLS] token itself
        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def generate_rollout(self, input, method_name="rollout"):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        cams_text = []
        cams_image = []
        # language self attention
        blocks = model.lxmert.encoder.layer
        for blk in blocks:
            cam = blk.attention.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)


        # image self attention
        blocks = model.lxmert.encoder.r_layers
        for blk in blocks:
            cam = blk.attention.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break

            # language self attention
            cam = blk.lang_self_att.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)

            # image self attention
            cam = blk.visn_self_att.self.get_attn().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn cam will be the one used for the R_t_i matrix
        cam_t_i = blk.visual_attention.att.get_attn().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_t = compute_rollout_attention(copy.deepcopy(cams_text))
        self.R_i_i = compute_rollout_attention(cams_image)
        self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))
        # language self attention
        cam = blk.lang_self_att.self.get_attn().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        cams_text.append(cam)

        self.R_t_t = compute_rollout_attention(cams_text)

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i