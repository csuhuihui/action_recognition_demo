import time
import sys
sys.path.append('.')
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from demo_dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import cv2
import time


global RGBnet
global Flownet
num_class = 11
test_segments = 12     # frame number
test_crops = 10

def load_net(RGBweights, Flowweights):	
	# weights: model weight

	global RGBnet
	global Flownet
	global num_class

	#******************* load RGB Net **********************
	print('Loading RGB Net......')
	RGBnet = TSN(num_class, 1, 'RGB',
			  base_model='BNInception',
			  consensus_type='avg',
			  dropout=0.7)

	checkpoint = torch.load(RGBweights)
	base_dict_RGB = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
	RGBnet.load_state_dict(base_dict_RGB)
	print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

	#******************* load RGB Net **********************
	print('Loading Flow Net......')
	Flownet = TSN(num_class, 1, 'Flow',
			  base_model='BNInception',
			  consensus_type='avg',
			  dropout=0.7)

	checkpoint = torch.load(Flowweights)
	print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
	
	base_dict_Flow = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
	Flownet.load_state_dict(base_dict_Flow)


def slip_video(video_file,save_path):
	count = 0
	cap = cv2.VideoCapture(video_file)
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	ret, pre_frame = cap.read()
	prvs = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
	while True:
		flag, frame = cap.read()
		if flag:
			count = count+1
			rgb = save_path+"/"+"img_%05d.jpg" %(count)
			flow_x = save_path+"/"+"flow_x_%05d.jpg"%(count)
			flow_y = save_path+"/"+"flow_y_%05d.jpg"%(count)
			next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			cv2.normalize(flow, flow, 0, 255, cv2.NORM_MINMAX)
			prvs = next
			cv2.imwrite(rgb,frame)
			cv2.imwrite(flow_x,flow[:,:,0])
			cv2.imwrite(flow_y, flow[:,:,1])
		else:
			break


def test_models(fileAddr, gpus):
	# fileAddr: video address
	# frames: total frame number of the video
	# gpus: gpu index

	global RGBnet
	global Flownet
	global num_class
	global test_segments
	global test_crops
	devices = [gpus]


	# ****************************** Read Camera ************************************
	cap = cv2.VideoCapture(0)
	fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	clip = []
	#saving video
	w = int(cap.get(3))
	h = int(cap.get(4))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (w,h),1)
	start = time.time()
	while True:
		ret, frame = cap.read()	
		if ret:			
			tmp = frame
			clip.append(cv2.resize(tmp, (171, 128)))
			if len(clip) == 20:
				inputs = np.array(clip).astype(np.float32)
				inputs = np.expand_dims(inputs, axis=0)
				inputs = inputs[:,:,8:120,30:142,:]
				inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
				clip.pop(0)
				out.write(frame)
				cv2.imshow('result', frame)
				cv2.waitKey(2)
		if time.time() - start > 10:
			break
	cap.release()
    out.release()
	cv2.destroyAllWindows()



	# ************************ Get RGB and Flow Image *******************************
	print('s'*20)
	slip_video('output.avi','output')
	print('e'*20)
	frames = '200'


	

	# **************************** Get RGB Result ***********************************
	
	cropping = torchvision.transforms.Compose([GroupOverSample(RGBnet.input_size, RGBnet.scale_size)])
	
	data_loader = torch.utils.data.DataLoader(
			TSNDataSet("", fileAddr, num_frames = frames, num_segments=test_segments,
					   new_length=1,
					   modality='RGB',
					   image_tmpl="img_{:05d}.jpg",
					   test_mode=True,
					   transform=torchvision.transforms.Compose([
						   cropping,
						   Stack(roll='BNInception' == 'BNInception'),
						   ToTorchFormatTensor(div='BNInception' != 'BNInception'),
						   GroupNormalize(RGBnet.input_mean, RGBnet.input_std),
					   ])),
			batch_size=1, shuffle=False,
			num_workers=2, pin_memory=True)


	RGBnet = torch.nn.DataParallel(RGBnet.cuda(devices[0]), device_ids=devices)
	RGBnet.eval()

	data_gen = enumerate(data_loader)

	total_num = len(data_loader.dataset)
	RGBoutput = []


	def eval_video_RGB(video_data):
		i, data, label = video_data
		num_crop = test_crops
		input_var = torch.autograd.Variable(data.view(-1, 3, data.size(2), data.size(3)),
											volatile=True)
		rst = RGBnet(input_var).data.cpu().numpy().copy()
		return i, rst.reshape((num_crop, test_segments, num_class)).mean(axis=0).reshape(
			(test_segments, 1, num_class)
		), label[0]


	proc_start_time = time.time()
	max_num = len(data_loader.dataset)

	for i, (data, label) in data_gen:
		if i >= max_num:
			break
		rst = eval_video_RGB((i, data, label))
		RGBoutput.append(rst[1:])
	video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in RGBoutput]
	print([(np.mean(x[0], axis=0)) for x in RGBoutput])
	video_labels = [x[1] for x in RGBoutput]

	print('RGBLabel: ' + str(video_pred))

	# reorder before saving
	name_list = fileAddr.strip().split()   ## video address

	order_RGBdict = {e:i for i, e in enumerate(sorted(name_list))}

	reorder_RGBoutput = [None] * len(RGBoutput)
	reorder_RGBlabel = [None] * len(RGBoutput)

	for i in range(len(RGBoutput)):
		idx = order_RGBdict[name_list[i]]
		reorder_RGBoutput[idx] = RGBoutput[i]
		reorder_RGBlabel[idx] = video_labels[i]

	RGBscores=reorder_RGBoutput
	RGBlabels=reorder_RGBlabel
	
		
	# **************************** Get Flow Result ***********************************
	
	cropping = torchvision.transforms.Compose([GroupOverSample(Flownet.input_size, Flownet.scale_size)])
	
	data_loader = torch.utils.data.DataLoader(
			TSNDataSet("", fileAddr, num_frames=frames, num_segments=test_segments,
					   new_length=5,
					   modality='Flow',
					   image_tmpl="flow_"+"{}_{:05d}.jpg",
					   test_mode=True,
					   transform=torchvision.transforms.Compose([
						   cropping,
						   Stack(roll='BNInception' == 'BNInception'),
						   ToTorchFormatTensor(div='BNInception' != 'BNInception'),
						   GroupNormalize(Flownet.input_mean, Flownet.input_std),
					   ])),
			batch_size=1, shuffle=False,
			num_workers=2, pin_memory=True)


	Flownet = torch.nn.DataParallel(Flownet.cuda(devices[0]), device_ids=devices)
	Flownet.eval()

	data_gen = enumerate(data_loader)

	total_num = len(data_loader.dataset)
	Flowoutput = []


	def eval_video_Flow(video_data):
		i, data, label = video_data
		num_crop = test_crops
		length = 10
		input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
											volatile=True)
		rst = Flownet(input_var).data.cpu().numpy().copy()
		return i, rst.reshape((num_crop, test_segments, num_class)).mean(axis=0).reshape(
			(test_segments, 1, num_class)
		), label[0]


	proc_start_time = time.time()
	max_num = len(data_loader.dataset)

	for i, (data, label) in data_gen:
		if i >= max_num:
			break
		rst = eval_video_Flow((i, data, label))
		Flowoutput.append(rst[1:])
		
	video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in Flowoutput]

	video_labels = [x[1] for x in Flowoutput]

	print('FlowLabel: ' + str(video_pred))
	print([(np.mean(x[0], axis=0)) for x in Flowoutput])

	order_dict = {e:i for i, e in enumerate(sorted(name_list))}

	reorder_Flowoutput = [None] * len(Flowoutput)
	reorder_Flowlabel = [None] * len(Flowoutput)

	for i in range(len(Flowoutput)):
		idx = order_dict[name_list[i]]
		reorder_Flowoutput[idx] = Flowoutput[i]
		reorder_Flowlabel[idx] = video_labels[i]

	Flowscores=reorder_Flowoutput
	FLowlabels=reorder_Flowlabel
	
	
	#***************************** Score Aggregation **********************************
	
	def softmax(raw_score, T=1):
		exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
		sum_s = exp_s.sum(axis=-1)
		return exp_s / sum_s[..., None]

	def default_aggregation_func(score_arr, normalization=True, crop_agg=None):
		"""
		This is the default function for make video-level prediction
		:param score_arr: a 3-dim array with (frame, crop, class) layout
		:return:
		"""
		crop_agg = np.mean if crop_agg is None else crop_agg
		if normalization:
			return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
		else:
			return crop_agg(score_arr, axis=1).mean(axis=0)
			
	def eval_scores(score_weights = [1.5,1],crop_agg = 'mean'):
		score_list = [np.array(Flowscores[0][0]), np.array(RGBscores[0][0])]
		agg_score_list = []
		for score_vec in score_list:

			agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np,crop_agg)) for x in score_vec]
			agg_score_list.append(np.array(agg_score_vec))

		final_scores = np.zeros_like(agg_score_list[0])
		for i, agg_score in enumerate(agg_score_list):
			final_scores += agg_score * score_weights[i]
		print('Final score:' + str(final_scores))
		final_scores = list(final_scores)
		label_index = final_scores.index(max(final_scores))
		label_dic = ['ApplyEyeMakeup','ApplyLipstick','BlowDryHair','BlowingCandles','BodyWeightSquats','BrushingTeeth','HeadMassage','JumpRope','TaiChi','Typing','WritingOnBoard']
		print(label_index)
		print(label_dic[label_index])
	eval_scores()
	
	
	
	
	
	
#load_net('opencv_ucf101_bninception__rgb_model_best.pth.tar', 'opencv_ucf101_bninception__flow_model_best.pth.tar')
load_net('rgb_model_best.pth.tar','flow_model_best.pth.tar')
test_models('output', 0)
