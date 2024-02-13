import cv2
import numpy as np

from python.infer import get_test_images
from python.preprocess import preprocess, NormalizeImage, Permute, Resize_Mult32
from vehicle_plateutils import create_predictor, get_infer_gpuid, get_rotate_crop_image
from vehicleplate_postprocess import build_post_process
from paddle import inference

class PlateDetector(object):
    def __init__(self, model_path):
        self.pre_process_list = {
            'Resize_Mult32': {
                'limit_side_len': 736,
                'limit_type': 'min',
            },
            'NormalizeImage': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'is_scale': True,
            },
            'Permute': {}
        }
        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 100
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"

        self.postprocess_op = build_post_process(postprocess_params)

        model_file_path = '{}/{}.pdmodel'.format(model_path, 'inference')
        params_file_path = '{}/{}.pdiparams'.format(model_path, 'inference')
    
        self.config = inference.Config(model_file_path, params_file_path)
        self.config.enable_use_gpu(500, 0)
        self.config.enable_memory_optim()
        self.config.disable_glog_info()
        self.config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        self.config.delete_pass("matmul_transpose_reshape_fuse_pass")
        self.config.switch_use_feed_fetch_ops(False)
        self.config.switch_ir_optim(True)

        self.predictor = inference.create_predictor(self.config)
        
        input_names = self.predictor.get_input_names()

        for name in input_names:
            self.input_tensor = self.predictor.get_input_handle(name)

        output_names = self.predictor.get_output_names()
        self.output_tensors = []
        output_name = 'softmax_0.tmp_0'
    
        if output_name in output_names:
            return [self.predictor.get_output_handle(output_name)]
    
        else:
            for output_name in output_names:
                output_tensor = self.predictor.get_output_handle(output_name)
                self.output_tensors.append(output_tensor)

    def preprocess(self, im_path):
        preprocess_ops = []
        for op_type, new_op_info in self.pre_process_list.items():
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []

        im, im_info = preprocess(im_path, preprocess_ops)
        input_im_lst.append(im)
        input_im_info_lst.append(im_info['im_shape'] / im_info['scale_factor'])

        return np.stack(input_im_lst, axis=0), input_im_info_lst

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, image):
        
        img, shape_list = self.preprocess(image)
        if img is None:
            return None, 0
        self.input_tensor.copy_from_cpu(img)
            
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['maps'] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)

        org_shape = image.shape
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, org_shape)
        if dt_boxes.shape[0]>1:
            dt_boxes = dt_boxes[0]
        
        xmin1,ymin1,xmax1,ymin2,xmax2,ymax1,xmin2,ymax2 = dt_boxes.reshape(8,).astype(np.int)
        
        xmin = min(xmin1, xmin2)
        ymin = min(ymin1, ymin2)
        xmax = max(xmax1, xmax2)
        ymax = max(ymax1,ymax2)
        return [xmin, ymin, xmax, ymax]
