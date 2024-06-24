from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile

from utils.utils import InputPadder, compute_out_of_boundary_mask
from glob import glob
from gmflow.geometry import forward_backward_consistency_check
from gmflow.gmflow import GMFlow
import cv2
from matplotlib import pyplot as plt

class gmflow_interface:

	def __init__(self):

		torch.backends.cudnn.benchmark = True
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		local_rank = 0
		feature_channels = 128
		num_scales = 2
		upsample_factor = 4
		num_head = 1
		attention_type = 'swin'
		ffn_dim_expansion = 4
		num_transformer_layers = 6
		resume = "/home/mverghese/MBLearning/gmflow/pretrained/gmflow_with_refine_things-36579974.pth"
		# resume = "/home/mverghese/MBLearning/gmflow/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"

		strict_resume = True
		self.model = GMFlow(feature_channels=feature_channels,
					num_scales=num_scales,
					upsample_factor=upsample_factor,
					num_head=num_head,
					attention_type=attention_type,
					ffn_dim_expansion=ffn_dim_expansion,
					num_transformer_layers=num_transformer_layers,
					).to(device)


		start_epoch = 0
		start_step = 0
		# resume checkpoints
		print('Load checkpoint: %s' % resume)

		loc = 'cuda:{}'.format(local_rank)
		checkpoint = torch.load(resume, map_location=loc)

		weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

		self.model.load_state_dict(weights, strict=strict_resume)

		print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

	def inference(self,frame_1,frame_2):
		padding_factor = 32
		attn_splits_list = [2, 8]
		corr_radius_list = [-1, 4]
		prop_radius_list = [-1, 1]
		inference_size = None
		paired_data = False
		pred_bidir_flow = False
		fwd_bwd_consistency_check = False

		self.model.eval()

		stride = 1


		

		image1 = frame_1.astype(np.uint8)
		image2 = frame_2.astype(np.uint8)

		if len(image1.shape) == 2:  # gray image, for example, HD1K
			image1 = np.tile(image1[..., None], (1, 1, 3))
			image2 = np.tile(image2[..., None], (1, 1, 3))
		else:
			image1 = image1[..., :3]
			image2 = image2[..., :3]

		image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
		image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

		padder = InputPadder(image1.shape, padding_factor=padding_factor)
		image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())


		# resize before inference

		results_dict = self.model(image1, image2,
							 attn_splits_list=attn_splits_list,
							 corr_radius_list=corr_radius_list,
							 prop_radius_list=prop_radius_list,
							 pred_bidir_flow=pred_bidir_flow,
							 )

		flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

		# resize back

		flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 2]
		return(flow)

	def save_vis(self,flow,output_path):
		save_vis_flow_tofile(flow, output_path)


if __name__ == '__main__':
	im_1 = cv2.imread("demo/davis_breakdance-flare/00000.jpg")
	im_2 = cv2.imread("demo/davis_breakdance-flare/00001.jpg")

	im_1 = cv2.cvtColor(im_1,cv2.COLOR_BGR2RGB)
	im_2 = cv2.cvtColor(im_2,cv2.COLOR_BGR2RGB)

	interface = gmflow_interface()
	flow = interface.inference(im_1,im_2)
	plt.imshow(np.hstack((flow[:,:,0],flow[:,:,1])))
	plt.show()

	interface.save_vis(flow,"out.jpg")

	
