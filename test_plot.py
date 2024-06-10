dsm_i_pos = [66.51, 58.25, 53.97, 49.16, 47.73, 45.26, 43.04, 38.63, 39.79]
partial_lrp_i_pos = [66.62, 58.53, 53.18, 46.87, 46.32, 44.74, 42.35, 37.63, 39.79]
# partial_lrp_i_pos = [66.51, 57.38, 53.3, 48.18, 47.11, 44.65, 42.25, 37.96, 39.79]

diff1 = []
diff2 = []


for i in range(0, 8):
    if i < 3:
        diff1.append( 0.25 * (dsm_i_pos[i] - dsm_i_pos[i + 1]) )
    else:
        diff1.append( 0.05 * (dsm_i_pos[i] - dsm_i_pos[i + 1]) )

for i in range(1, 8):
    if i < 3:
        diff2.append( 0.25 * (partial_lrp_i_pos[i] - partial_lrp_i_pos[i + 1]) )
    else:
        diff2.append( 0.05 * (partial_lrp_i_pos[i] - partial_lrp_i_pos[i + 1]) )


print(f"LRP: {sum(diff2)/8}, DSM: {sum(diff1)/8}")