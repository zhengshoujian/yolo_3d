import cv2
import numpy as np
import os
import json
from util.post_processing import gen_3D_box, draw_3D_box, draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Construct the network
class det_3d(object):
    def __init__(self, config):
        self.calib_file = config["dir"]["calib_file"]
        self.box2d_dir = config["dir"]["box2d_dir"]
        self.model = bbox_3D_net((224, 224, 3))
        self.model.load_weights(config["model_dir"])

        self.classes = config["class"]
        self.cls_to_ind = {cls: i for i, cls in enumerate(self.classes)}

        self.dims_avg_dir = dir["dims_avg"]
        self.dims_avg = np.loadtxt(self.dims_avg_dir, delimiter=',')

        self.cam_to_img = get_cam_data(self.calib_file)
        self.fx = self.cam_to_img[0][0]
        self.u0 = self.cam_to_img[0][2]
        self.v0 = self.cam_to_img[1][2]

    def get_3d(self, f):
        image_file = self.image_dir + f
        box2d_file = self.box2d_dir + f.replace('png', 'txt')
        img = cv2.imread(image_file)
        dect2D_data, box2d_reserved = get_dect2D_data(box2d_file, self.classes)
        for data in dect2D_data:
            cls = data[0]
            box_2D = np.asarray(data[1],dtype=np.float)
            xmin = box_2D[0]
            ymin = box_2D[1]
            xmax = box_2D[2]
            ymax = box_2D[3]

            patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            patch = cv2.resize(patch, (224, 224))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]])
            patch = np.expand_dims(patch, 0)

            prediction = self.model.predict(patch)

            # compute dims
            dims = self.dims_avg[self.cls_to_ind[cls]] + prediction[0][0]

            # Transform regressed angle
            box2d_center_x = (xmin + xmax) / 2.0
            # Transfer arctan() from (-pi/2,pi/2) to (0,pi)
            theta_ray = np.arctan(self.fx /(box2d_center_x - self.u0))
            if theta_ray<0:
                theta_ray = theta_ray+np.pi

            max_anc = np.argmax(prediction[2][0])
            anchors = prediction[1][0][max_anc]

            if anchors[1] > 0:
                angle_offset = np.arccos(anchors[0])
            else:
                angle_offset = -np.arccos(anchors[0])

            bin_num = prediction[2][0].shape[0]
            wedge = 2. * np.pi / bin_num
            theta_loc = angle_offset + max_anc * wedge

            theta = theta_loc + theta_ray
        # object's yaw angle
            yaw = np.pi/2 - theta

            points2D = gen_3D_box(yaw, dims, self.cam_to_img, box_2D)
        
        return points2D, box2d_reserved, img


if __name__ == "__main__":
    f = open("config.json")
    config = json.loads(f)
    image_dir = config["dir"]["image_dir"]
    all_image = sorted(os.listdir(image_dir))
    detector = det_3d(config)
    for f in all_image:
        points2D, box2d_reserved, img = detector.get_3d(f)
        draw_3D_box(img, points2D)
        for cls, box in box2d_reserved:
            draw_2D_box(img, box)
        cv2.imshow(f, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite('output/'+ f.replace('png','jpg'), img)
