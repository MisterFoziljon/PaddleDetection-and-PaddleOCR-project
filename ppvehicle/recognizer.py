from ppocr.postprocess import build_post_process
from paddle import inference
import cv2
import math
import numpy as np
import time

class Recognizer():
    def __init__(self,model_path, dict_path):
        self.rec_image_shape = [3, 48, 320]
        self.rec_algorithm = 'SVTR_LCNet'
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path":dict_path,
            "use_space_char": True
        }
        
        model_file_path = '{}/{}.pdmodel'.format(model_path, 'inference')
        params_file_path = '{}/{}.pdiparams'.format(model_path, 'inference')
        
        self.postprocess_op = build_post_process(postprocess_params)
        self.config = inference.Config(model_file_path, params_file_path)
        self.config.enable_use_gpu(500, 0) # gpu_mem, gpu_id
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
            self.output_tensors = [self.predictor.get_output_handle(output_name)]
        
        else:
            for output_name in output_names:
                output_tensor = self.predictor.get_output_handle(output_name)
                self.output_tensors.append(output_tensor)
        
        
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, image):        
        rec_res = [['', 0.0]]
        st = time.time()
        norm_img_batch = []
        imgC, imgH, imgW = self.rec_image_shape[:3]
        max_wh_ratio = imgW / imgH
        h, w = image.shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        norm_img = self.resize_norm_img(image, max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch = norm_img.copy()
    
        self.input_tensor.copy_from_cpu(norm_img_batch)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        
        self.predictor.try_shrink_memory()
        rec_result = self.postprocess_op(outputs[0])
        
        return rec_result[0]+(time.time()-st,)