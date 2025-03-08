from retrieve_salience import wrapper_mask


saliency_list = wrapper_mask()

for i, sal_map in enumerate(saliency_list):
    print(f"Saliency #{i} shape: {sal_map.shape}")

