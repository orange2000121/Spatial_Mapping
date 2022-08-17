from openni import openni2
import numpy as np
import cv2
# import depth_to_point_cloud as d2p

class AsusXtion2():
    def __init__(self):
        # ----------------------------- camera intrinsic ----------------------------- #
        self.w = 480
        self.h = 640
        self.fx = 250
        self.fy = 250
        self.cx = self.w/2
        self.cy = self.h/2
        # -------------------------------- set camera -------------------------------- #
        openni2.initialize()
        self.dev = openni2.Device.open_any()
        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream = self.dev.create_color_stream()
        self.depth_stream.start()
        self.color_stream.start()
        # self.cap = cv2.VideoCapture(0)
        # ----------------------------------- info ----------------------------------- #
        print(self.dev.get_device_info())

    def mousecallback(self,event,x,y,flags,param):
        '''
        When mouse is clicked, print the depth value at the clicked position
        '''
        if event==cv2.EVENT_LBUTTONDBLCLK:
            print(y, x, dpt[y,x])
    def get_intrinsic(self):
        '''
        Get intrinsic matrix of the Asus Xtion 2.
        Returns:
            camera intrinsic matrix
        '''
        return self.dev.get_sensor_info()

    def read_rgb_depth_image(self,show=False):
        '''
        Read single RGB and Depth image from Asus Xtion 2.
        returns:
            Any rgb image
            NDArray depth image
        '''
        frame = self.depth_stream.read_frame()
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        
        dpt2 *= 25
        dpt = dpt1 + dpt2

        # ret, rgb = self.cap.read()
        frame = self.color_stream.read_frame()
        rgb = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 3])[:, :, ::-1]

        if show:
            cv2.namedWindow('depth')
            cv2.setMouseCallback('depth',self.mousecallback)
            cv2.imshow('depth', dpt)
            cv2.imshow('color', rgb)
            cv2.imwrite('img/depth.png', dpt)
            cv2.imwrite('img/color.png', rgb)
            cv2.waitKey(1000)

        return rgb, dpt

    def __del__(self):
        print('close')
        self.depth_stream.stop()
        self.dev.close()

# if __name__ == "__main__": 

    # xtion = AsusXtion2() # create depth camera object
    # print(xtion.get_intrinsic()) # get intrinsic matrix of the depth camera
    # while True:
    #     frame, dpt = xtion.read_rgb_depth_image() # read rgb and depth image
    #     # d2p.depth_to_point_cloud(frame, dpt, camera_intrinsic) # convert depth image to point cloud
    #     cv2.imshow('depth', dpt)    # show depth image
    #     cv2.imshow('color', frame)  # show color image

    #     key = cv2.waitKey(1)
    #     if int(key) == ord('q'):
    #         break
