import os
import yaml
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from preprocess import preprocess, Resize, Permute, PadStride, LetterBoxResize, WarpAffine, Pad, decode_image, CULaneResize

def create_inputs(imgs, im_info):
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs

class PredictConfig():
    def __init__(self, model_dir, use_fd_format=False):
        fd_deploy_file = os.path.join(model_dir, 'inference.yml')
        ppdet_deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        deploy_file = ppdet_deploy_file
        
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False

class Detector(object):
    def __init__(self, model_dir):
        paddle.enable_static()
        self.pred_config = self.set_config(model_dir, use_fd_format=False)
            
        infer_model = '{}/{}.pdmodel'.format(model_dir, 'inference')
        infer_params = '{}/{}.pdiparams'.format(model_dir, 'inference')
       
        self.config = Config(infer_model, infer_params)
        self.config.enable_use_gpu(200, 0)
        self.config.switch_ir_optim(True)
    
        self.config.disable_glog_info()
        self.config.enable_memory_optim()
    
        self.config.switch_use_feed_fetch_ops(False)
        self.predictor = create_predictor(self.config)
        
        self.batch_size = 1

    def set_config(self, model_dir, use_fd_format):
        return PredictConfig(model_dir, use_fd_format=False)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def predict(self, repeats=1, run_benchmark=False):
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None

        if run_benchmark:
            for i in range(repeats):
                self.predictor.run()
                paddle.device.cuda.synchronize()
            result = dict(
                boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
            return result

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num' 
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        
        return result
        
    def __call__(self, image_list):
        # preprocess
        inputs = self.preprocess(image_list)
        # model prediction
        result = self.predict()
        # postprocess
        result = self.postprocess(inputs, result)

        im_bboxes_num = result['boxes_num'][0]
        
        np_boxes = result['boxes'][0:0 + im_bboxes_num, :]
        labels = self.pred_config.labels
        
        expect_boxes = (np_boxes[:, 1] > 0.7) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]

        xmin, ymin, xmax, ymax = None, None, None, None
        results = [dt[2:].astype(np.int32) for dt in np_boxes]
        return results