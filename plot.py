import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
x = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


# HILA CHEFER RELEVANCE MAPS:
hcrm_i_pos = [66.51, 55.42, 49.96, 46.13, 45.29, 43.67, 41.81, 37.9, 39.79]
hcrm_i_neg = [66.51, 66.64, 66.42, 63.27, 60.72, 58.54, 53.75, 43.91, 39.79]
hcrm_t_pos = [66.51, 29.56, 14.46, 7.07, 6.23, 6.12, 4.82, 4.4, 4.37]
hcrm_t_neg = [66.51, 64.31, 59.43, 39.66, 28.3, 21.07, 6.23, 4.39, 4.37]

# TRANSFORMER ATTR:
transformer_attr_i_pos = [66.51, 57.38, 53.3, 48.18, 47.11, 44.65, 42.25, 37.96, 39.79]
transformer_attr_i_neg = [66.62, 65.96, 63.63, 60.11, 58.53, 55.75, 51.56, 42.56, 39.79]
transformer_attr_t_pos = [66.62, 30.99, 15.31, 6.21, 4.79, 4.8, 4.74, 4.52, 4.52]
transformer_attr_t_neg = [66.62, 64.66, 58.2, 38.19, 27.13, 20.56, 6.17, 4.54, 4.52]

# RAW ATTN
raw_attn_i_pos = [66.51, 60.12, 55.38, 49.32, 47.79, 45.56, 43.92, 38.31, 39.79]
raw_attn_i_neg = [66.51, 66.58, 64.94, 60.48, 58.0, 54.8, 48.79, 39.91, 39.79]
raw_attn_t_pos = [66.51, 52.95, 29.96, 14.53, 11.0, 8.75, 4.7, 4.4, 4.37]
raw_attn_t_neg = [66.51, 54.0, 38.04, 24.47, 19.37, 16.27, 5.92, 4.4, 4.37]

# PARTIAL LRP
partial_lrp_i_pos = [66.62, 58.53, 53.18, 46.87, 46.32, 44.74, 42.35, 37.63, 39.79]
partial_lrp_i_neg = [66.62, 66.01, 63.73, 58.71, 56.87, 54.65, 51.34, 42.31, 39.79]
partial_lrp_t_pos = [66.62, 34.74, 17.77, 9.3, 7.6, 6.72, 4.75, 4.55, 4.52]
partial_lrp_t_neg = [66.62, 59.84, 51.47, 36.05, 25.87, 20.17, 6.12, 4.55, 4.52]

# GRADCAM
gradcam_i_pos = [66.51, 63.6, 60.32, 55.14, 52.9, 49.94, 46.93, 38.49, 39.79]
gradcam_i_neg = [66.51, 65.31, 62.87, 57.36, 54.56, 51.55, 47.08, 39.53, 39.79]
gradcam_t_pos = [66.51, 51.31, 35.21, 18.85, 12.72, 10.25, 5.1, 4.37, 4.37]
gradcam_t_neg = [66.51, 51.97, 38.21, 27.42, 21.63, 17.74, 5.89, 4.4, 4.37]

# ROLLOUT
rollout_i_pos = [66.51, 62.25, 58.43, 54.14, 52.72, 50.84, 47.94, 39.52, 39.79]
rollout_i_neg = [66.51, 66.08, 61.78, 53.28, 50.26, 48.02, 44.92, 36.7, 39.79]
rollout_t_pos = [66.51, 58.46, 39.86, 24.05, 17.62, 14.47, 5.52, 4.39, 4.37]
rollout_t_neg = [66.51, 48.33, 30.1, 14.07, 9.46, 7.84, 4.92, 4.37, 4.37]


# MY OG DSM METHOD
dsm_i_pos = [66.51, 58.25, 53.97, 49.16, 47.73, 45.26, 43.04, 38.63, 39.79]
dsm_i_neg = [66.62, 65.59, 63.55, 55.46, 52.78, 48.48, 44.48, 38.3, 39.79]
dsm_t_pos = [66.51, 40.22, 26.51, 14.2, 10.95, 8.43, 4.85, 4.37, 4.37]
dsm_t_neg = [66.62, 55.38, 46.96, 27.44, 18.53, 14.16, 5.56, 4.56, 4.52]

# MY OG DSM METHOD + GRAD CONTRIBUTION (cumulative sum)
# dsm_grad_i_pos = [66.62, 58.02, 54.11, 50.17, 48.21, 46.94, 44.0, 38.73, 39.79]
dsm_grad_i_pos = [66.62, 57.83, 53.79, 49.01, 48.16, 46.88, 43.62, 38.25, 39.79]
dsm_grad_i_neg = [66.62, 66.83, 65.47, 61.62, 59.99, 57.54, 52.91, 42.4, 39.79]
dsm_grad_t_pos = [66.62, 34.08, 20.63, 9.81, 7.61, 6.9, 4.8, 4.55, 4.52]
dsm_grad_t_neg = [66.62, 61.73, 53.83, 33.24, 22.32, 17.16, 5.7, 4.55, 4.52]



plt.title('Positive perturbation test on image modality')
plt.plot(x, dsm_i_pos)
plt.plot(x, dsm_grad_i_pos)
plt.plot(x, hcrm_i_pos)
plt.plot(x, transformer_attr_i_pos)
plt.plot(x, raw_attn_i_pos)
plt.plot(x, partial_lrp_i_pos)
plt.plot(x, gradcam_i_pos)
plt.plot(x, rollout_i_pos)



plt.legend(['DSM (' + str(round(sum(dsm_i_pos)/9, 2)) + ')', 
            'DSM + Grad (' + str(round(sum(dsm_grad_i_pos)/9, 2)) + ')', 
            'Relevance maps (' + str(round(sum(hcrm_i_pos)/9, 2)) + ')', 
            'Transformer attribution (' + str(round(sum(transformer_attr_i_pos)/9, 2)) + ')', 
            'Raw attention (' + str(round(sum(raw_attn_i_pos)/9, 2)) + ')', 
            'LRP (' + str(round(sum(partial_lrp_i_pos)/9, 2)) + ')',
            'GradCAM (' + str(round(sum(gradcam_i_pos)/9, 2)) + ')', 
            'Rollout (' + str(round(sum(rollout_i_pos)/9, 2)) + ')'
          ]) 


# plt.title('Negative perturbation test on image modality')
# plt.plot(x, dsm_i_neg)
# plt.plot(x, dsm_grad_i_neg)
# plt.plot(x, hcrm_i_neg)
# plt.plot(x, transformer_attr_i_neg)
# plt.plot(x, raw_attn_i_neg)
# plt.plot(x, partial_lrp_i_neg)
# plt.plot(x, gradcam_i_neg)
# plt.plot(x, rollout_i_neg)



# plt.legend([ 'DSM (' + str(round(sum(dsm_i_neg)/9, 2)) + ')', 
#             'DSM + Grad (' + str(round(sum(dsm_grad_i_neg)/9, 2)) + ')', 
#             'Relevance maps (' + str(round(sum(hcrm_i_neg)/9, 2)) + ')', 
#             'Transformer attribution (' + str(round(sum(transformer_attr_i_neg)/9, 2)) + ')', 
#             'Raw attention (' + str(round(sum(raw_attn_i_neg)/9, 2)) + ')', 
#             'LRP (' + str(round(sum(partial_lrp_i_neg)/9, 2)) + ')',
#             'GradCAM (' + str(round(sum(gradcam_i_neg)/9, 2)) + ')', 
#             'Rollout (' + str(round(sum(rollout_i_neg)/9, 2)) + ')'
#           ]) 



# plt.title('Positive perturbation test on text modality')
# plt.plot(x, dsm_t_pos)
# plt.plot(x, dsm_grad_t_pos)
# plt.plot(x, hcrm_t_pos)
# plt.plot(x, transformer_attr_t_pos)
# plt.plot(x, raw_attn_t_pos)
# plt.plot(x, partial_lrp_t_pos)
# plt.plot(x, gradcam_t_pos)
# plt.plot(x, rollout_t_pos)



# plt.legend([ 'DSM (' + str(round(sum(dsm_t_pos)/9, 2)) + ')', 
#             'DSM + Grad (' + str(round(sum(dsm_grad_t_pos)/9, 2)) + ')', 
#             'Relevance maps (' + str(round(sum(hcrm_t_pos)/9, 2)) + ')', 
#             'Transformer attribution (' + str(round(sum(transformer_attr_t_pos)/9, 2)) + ')', 
#             'Raw attention (' + str(round(sum(raw_attn_t_pos)/9, 2)) + ')', 
#             'LRP (' + str(round(sum(partial_lrp_t_pos)/9, 2)) + ')',
#             'GradCAM (' + str(round(sum(gradcam_t_pos)/9, 2)) + ')', 
#             'Rollout (' + str(round(sum(rollout_t_pos)/9, 2)) + ')'
#           ]) 


# plt.title('Negative perturbation test on text modality')
# plt.plot(x, dsm_t_neg)
# plt.plot(x, dsm_grad_t_neg)
# plt.plot(x, hcrm_t_neg)
# plt.plot(x, transformer_attr_t_neg)
# plt.plot(x, raw_attn_t_neg)
# plt.plot(x, partial_lrp_t_neg)
# plt.plot(x, gradcam_t_neg)
# plt.plot(x, rollout_t_neg)



# plt.legend([ 'DSM (' + str(round(sum(dsm_t_neg)/9, 2)) + ')', 
#             'DSM + Grad (' + str(round(sum(dsm_grad_t_neg)/9, 2)) + ')', 
#             'Relevance maps (' + str(round(sum(hcrm_t_neg)/9, 2)) + ')', 
#             'Transformer attribution (' + str(round(sum(transformer_attr_t_neg)/9, 2)) + ')', 
#             'Raw attention (' + str(round(sum(raw_attn_t_neg)/9, 2)) + ')', 
#             'LRP (' + str(round(sum(partial_lrp_t_neg)/9, 2)) + ')',
#             'GradCAM (' + str(round(sum(gradcam_t_neg)/9, 2)) + ')', 
#             'Rollout (' + str(round(sum(rollout_t_neg)/9, 2)) + ')'
#           ]) 


plt.xlabel('Fraction of tokens removed')
plt.ylabel('Accuracy')
plt.show()