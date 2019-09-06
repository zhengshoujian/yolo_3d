import cv2
import numpy as np
import os
import json
import yolo
from util.post_processing import gen_3D_box, draw_3D_box, draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class video(object):
    def __init__(self, config):
        path = config["video_path"]
        self.cap = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.out = cv2.VideoWriter('result_video/'+path.split("/")[-1], fourcc, fps, self.size)

    def get_image(self):
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                frame = cv2.flip(frame, 0)

                self.out.write(frame)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                else:
                    break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


class bbox_3(object):
    def __init__(self, dims=None, yaw=None, anchor=None):
        self.dims = dims
        self.yaw = yaw
        self.anchor = anchor


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

        self.bbox_3 = bbox_3()

    def get_3d(self, img, dets):
        result = []
        for det in dets:
            data = img[det[1:5]]
            self.bbox_3 = bbox_3()
            cls = det[0]

            patch = img[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
            patch = cv2.resize(patch, (224, 224))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]])
            patch = np.expand_dims(patch, 0)

            prediction = self.model.predict(patch)

            # compute dims
            self.bbox_3.dims = self.dims_avg[self.cls_to_ind[cls]] + prediction[0][0]

            # Transform regressed angle
            center = (det[0] + det[2]) / 2.0
            # Transfer arctan() from (-pi/2,pi/2) to (0,pi)

            theta_ray = np.arctan(self.fx / (center - self.u0))
            if theta_ray < 0:
                theta_ray = theta_ray + np.pi
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
            yaw = np.pi / 2 - theta
            self.bbox_3.yaw = yaw
            result.append(bbox_3)
        return result


if __name__ == "__main__":
    f = open("config.json")
    config = json.loads(f)
    image_dir = config["dir"]["image_dir"]
    all_image = sorted(os.listdir(image_dir))
    detector = det_3d(config)
    for image in all_image:
        dets = yolo.detctor(image)
        result = det_3d(image, dets)
        for res in result:
            points2D = gen_3D_box(res.yaw, res.dims, detector.cam_to_img)

