import numpy as np
import cv2,time,sys,os
import scipy.signal as signal
try:
    import ray # <-- For multiprocessing
    from utils import log, remaining_iter_ms
    from touch_signal_processing.nHHD.pynhhd import nHHD
    from touch_signal_processing.spectral import resize
except ModuleNotFoundError:
    import os.path, sys
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
    from utils import log
    from nHHD.pynhhd import nHHD
    from spectral import resize


try:
    import rospy
    from std_msgs.msg import Header
    from geometry_msgs.msg import WrenchStamped

    from sensor_msgs.msg import Image 
    from cv_bridge import CvBridge

    # from guriguri_bot.msg import vector_field

    from converter import numpy2f32multi

except ModuleNotFoundError:
    pass


""" 
To use the Digit (gelsight) sensor in Docker:
-----------------------------------
apt update && apt install udev
echo 'SUBSYSTEMS=="usb", ENV{DEVTYPE}=="usb_device", ATTRS{idVendor}=="2833", ATTR{idProduct}=="0209", GROUP="plugdev", MODE="0666"' >> /lib/udev/rules.d/50-DIGIT.rules
service udev restart
udevadm control --reload
udevadm trigger
# The sensor should be properly detected now.

Also:
pip install --upgrade opencv-python
"""
    

"""(A) MULTIPROCESSING-READY FUNCTION -----------------------------------------------------"""
def get(remote, info, variable:str, key=''):
    return ray.get(info.get.remote(variable, key)) if remote else info.get(variable, key)
def set(remote, info, variable:str, newdata, overwrite=False):
    return info.set.remote(variable, newdata, overwrite) if remote else info.set(variable, newdata, overwrite)

""" THREAD 1 - TOUCH SIGNAL PROCESSING """
@ray.remote(num_gpus=0.4)
def T1_touch(rate, touchinfo, robotinfo, remote=True, display=False):
    log('START', format=('purple','bold'))

    """ Touch sensor handler """
    setup_OpenCL()
    Ts = 1/rate
    measured = get(remote, touchinfo, 'params', 'meas_vars')
    finger = TouchSensor(output_vars=measured, sensor='DIGIT', equalizer='None')
    finger.get_ref_frame() 
    while True:
        t0 = time.time()
        vector_field, angle = finger.current_contact_state(display=display)
        set(remote, touchinfo, 'data', dict(zip(measured, vector_field)))
        set(remote, robotinfo, 'data', {'grasp_angle':angle})
        if cv2.waitKey(remaining_iter_ms(Ts, t0)) == 27: break   
    cv2.destroyAllWindows()
    finger.shut_down()


"""(C) MAIN CLASS -------------------------------------------------------------------------"""
class TouchSensor:
    """ Interface class with the digit sensor. Processing of touch data. """
    def __init__(self, equalizer:str="CLAHE", vf_res_scaledown=1/25,
                 output_vars = ('Fn', 'Tn'), sensor:str='DIGIT', frame_shape=(320,240)
            ):
        """
            - equalizer: ("BASIC", "CLAHE", "NONE") -> Method used to equalize the video stream.
            - vf_res_scaledown -> downscaling factor for the vector field resolution.
            - output_vars -> output variables
        """
        
        self.bridge = CvBridge()
        self.pub_img = rospy.Publisher('/tactile_image', Image, queue_size=10)
        self.pub_tw = rospy.Publisher("/touch_to_wrench", WrenchStamped, queue_size=10)
        self.pub_tw_2 = rospy.Publisher("/touch_to_wrench_2", WrenchStamped, queue_size=10)
        self.pub_fg = rospy.Publisher("/finger_vision", WrenchStamped, queue_size=10)
        self.pub_th = rospy.Publisher("/tactile_hhd", WrenchStamped, queue_size=10)
        # self.pub_vf = rospy.Publisher("/vector_field", vector_field, queue_size=10)
        self.pub_vf_img = rospy.Publisher('/vector_field_image', Image, queue_size=10)

        self.connect_to_sensor(sensor.upper())
        self._data = ('Fn', 'Ft', 'Tn', 'Tx', 'divergence', 'curl') # <-- Variables that can be output
        self._data = dict(zip(self._data, [None]*len(self._data)))
        equalizer = equalizer.upper() if isinstance(equalizer, str) else None
        self.equalize_ = cv2.equalizeHist if equalizer=="BASIC" else \
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply if equalizer=="CLAHE" \
            else None
        self.t = time.time()
        self.vf_res_scaledown=vf_res_scaledown
        self.out_vars = output_vars
        self.frame_shape = frame_shape if frame_shape is not None else self.sensor.get_frame().shape[:2]
        self.vf_grid_shape = (np.array(self.frame_shape)*self.vf_res_scaledown).astype('int')
        self.nhhd = nHHD(grid=tuple(self.vf_grid_shape), spacings=(1,1))
        self.get_contact_shape = ContactShape(draw_poly=False)

    def connect_to_sensor(self, sensor):
        """ Setup connection to sensor """
        if sensor=='DIGIT':
            from digit_interface import Digit, DigitHandler
            log("[ ]Attempting connection to Digit sensor...")
            while True:
                try:
                    sensors = DigitHandler.list_digits()
                    log(" • Detected Digit sensors: ", [f"{sensor['serial']}:{sensor['dev_name']}" for sensor in sensors])
                    if sensors                : break
                    time.sleep(1)
                except KeyboardInterrupt:
                    sys.exit()
            digit_id = sensors[0]['serial']
            self.sensor = Digit(digit_id)
            self.sensor.connect()
            self.sensor.set_resolution(Digit.STREAMS["QVGA"])
            self.sensor.set_fps(Digit.STREAMS["QVGA"]["fps"]["30fps"])
            log(f"[o] Connected to Digit sensor: {digit_id}")
        elif sensor == 'GS_MINI':
            from touch_signal_processing.gelsight import GelSight, Finger
            log("[ ]Attempting connection to GelSight-mini sensor...")
            cam_id = 0 # <-- change to 1 or 2 if you are getting your webcam stream
            self.sensor = GelSight(Finger.MINI, cam_id)
            self.sensor.connect()
            log(f"[o] Connected to camera in cam_id:{cam_id}.")
        else:
            self.sensor = Camera()         
    
    def shut_down(self):
        """Disconnect sensor safely"""
        self.sensor.disconnect()
        cv2.destroyAllWindows()

    """ Sensor stream handling """
    def equalize(self,frame):
        """Preprocessing of frames for improved quality"""
        if self.equalize_:
            eq = []
            frame = np.transpose(frame,(2,0,1))
            for channel in frame:
                eq.append(self.equalize_(channel))
            eq = cv2.merge(eq)
        else: eq = frame
        return eq

    def get_frame(self):
        """Get preprocessed frame from sensor streaming"""
        frame = self.sensor.get_frame()
        frame = resize(frame, self.frame_shape, inter=cv2.INTER_LINEAR)
        return self.equalize(frame)

    def get_ref_frame(self, save:bool=True, generate_new:bool=False, N:int=200):
        """Get reference frame for touch inference"""
        def create_ref_frame():
            avg = self.get_frame()
            for i in range(N):
                frame = self.get_frame()
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg = cv2.addWeighted(frame, alpha, avg, beta, 0.0)
            return avg  
        #---------
        if save:
            shape = self.get_frame().shape[:2]
            path = os.path.join(os.path.dirname(__file__), 'ref_frames')
            if not os.path.exists(path): os.mkdir(path)
            path = os.path.join(path, f'ref_{shape[0]}x{shape[1]}.npy')
            if generate_new or not os.path.isfile(path):
                log(f" [x] No saved ref_frame at [{os.path.relpath(path)}].", format='red', end='')
                log(f" --> Generating new ref_frame.", format='cyan', name_format=None)
                self.ref_frame = create_ref_frame()
                np.save(path, self.ref_frame)
                log(f" [o] Saved new ref_frame frame.", format='cyan')
            else:
                self.ref_frame = np.load(path)
                log(f" [o] Loaded ref_frame frame from [{os.path.relpath(path)}].", format='green')
        else:
            log(f" ... Generating ref_frame.", format='cyan')
            self.ref_frame = create_ref_frame()   
            log(f" [o] Generated ref_frame.", format='green')

    def get_diff_frame(self, frame):
        """Get differential frame"""
        return frame-self.ref_frame

    """ Computations """          
    def get_vector_field(self, frame, ref_frame):
        """Compute tactile vector field"""
        vfield = compute_optical_flow(ref_frame, frame)
        return resize(vfield, self.vf_grid_shape, inter=cv2.INTER_LINEAR)

    def get_field_center(self, field, rel_th=0.5, abs_th=0, sign=1):
        """Estimate virtual origin of scalar field (divergence/curl fields)"""
        field = sign*field
        rel_mask = field > np.percentile(field, rel_th*100)
        abs_mask = field > abs_th
        shape = field*rel_mask*abs_mask
        return get_center_of_mass(shape)

    def get_field_axis(self, field, center, rel_th=0.5, abs_th=0):
        """Estimate principal axis of scalar field (divergence/curl fields)"""
        rel_mask = field > np.percentile(field, rel_th*100)
        abs_mask = field > abs_th
        shape = field*rel_mask*abs_mask
        return get_orientation(shape, center)

    def decompose(self, vfield):
        """ nHHD decomposition of vector field: http://dx.doi.org/10.1109/TVCG.2014.2312012 """
        self.nhhd.decompose(vfield)

        # rotational center_plus corresponds to the minimum of the curl and vice versa
        rotational_center_plus = np.unravel_index(np.argmax(self.nhhd.nRu), self.nhhd.nRu.shape)
        rotational_center_minus = np.unravel_index(np.argmin(self.nhhd.nRu), self.nhhd.nRu.shape)

        # return self.nhhd.d, self.nhhd.r, self.nhhd.h
        return self.nhhd.d, self.nhhd.r, self.nhhd.h, rotational_center_plus, rotational_center_minus

    def current_contact_state(self, display=True, alpha=0.4):
        """ Main method to execute on every frame of Digit sensor video stream """
        for var in self.out_vars:
            assert var in self._data.keys(), f"Possible output variables are {self._data.keys()}."
        frame = self.get_frame()
        
        # TODO: account for time lag between sensor data and message publish
        stamp = rospy.Time.now()

        # convert numpy (CV::Mat) to image message. 
        # TODO: double check rgb vs bgr
        # imgMsg = self.bridge.cv2_to_imgmsg(frame, "rgb8")
        imgMsg = self.bridge.cv2_to_imgmsg(frame, "bgr8")

        # publish image topic in ros.
        self.publish_image(imgMsg, self.pub_img)

        vfield = self.get_vector_field(frame, self.ref_frame)
        vfield = self.vfield*alpha + vfield*(1-alpha) if hasattr(self, 'vfield') else vfield
        self.vfield = vfield

        vfield_msg = numpy2f32multi(self.vfield)
        
        # publish vector field message.
        # self.publish_vector_field(vfield_msg)

        # vd, vr, vh = self.decompose(vfield)
        vd, vr, vh, c_plus, c_minus = self.decompose(vfield)
        # Compute divergence and curl
        self._data['curl'] = get_curl(vr)
        self._data['divergence'] = get_divergence(vd)
        # Compute forces (proportional)
        # self._data['Ft'] = np.sum(vfield, axis=(0,1)) * 0.2
        # self._data['Fn'] = np.sum(self._data['divergence'])
        # self._data['Tn'] = np.sum(self._data['curl']) * 0.2
        # self._data['Tx'] = 0.0 # <-- not computed yet

        # fingervision FT calculation baseline
        Fz_array = np.sqrt(vfield[:,:,0] ** 2.0 + vfield[:,:,1]  ** 2.0)
        F_array = np.concatenate((vfield, Fz_array[:,:,None]), axis=2)

        index_mat = np.flip(np.stack(np.meshgrid(np.linspace(0,self.vf_grid_shape[1]-1,self.vf_grid_shape[1]), 
                                                 np.linspace(0,self.vf_grid_shape[0]-1,self.vf_grid_shape[0])), axis=2), axis=2)
        dist_from_center_array = np.zeros((self.vf_grid_shape[0], self.vf_grid_shape[1], 3))
        # note: flip indices 0 and 1 to ensure that dist_from_center_array has channels (x,y,z)
        # this is necessary for the cross product with F_array, having channels (x,y,z)
        dist_from_center_array[:,:,1] = index_mat[:,:,0] - (self.vf_grid_shape[0] - 1.0)/2.0
        dist_from_center_array[:,:,0] = index_mat[:,:,1] - (self.vf_grid_shape[1] - 1.0)/2.0
        tau_array = np.cross(dist_from_center_array, F_array)

        Fz_fingervision = np.sum(F_array[:,:,2]) / np.prod(self.vf_grid_shape)
        tau_x_fingervision = np.sum(tau_array[:,:,0]) / np.prod(self.vf_grid_shape)
        tau_y_fingervision = np.sum(tau_array[:,:,1]) / np.prod(self.vf_grid_shape)
        tau_z_fingervision = np.sum(tau_array[:,:,2]) / np.prod(self.vf_grid_shape)

        # copy+pasted and modified code from draw_flow to convert from vector to image coordinates
        [h, w] = self.frame_shape
        [hf, wf] = self.vf_grid_shape
        red_size_vf = self.frame_shape!=self.vf_grid_shape.shape[:2]
        (sh, sw) = (int(h/hf), int(w/wf)) if red_size_vf else (16, 16)
        y, x = np.mgrid[sh/2:h:sh, sw/2:w:sw].reshape(2,-1).astype(int)

        Fn_loc_no_thresh = self.get_field_center(self._data['divergence'], rel_th=0.0, abs_th=0.0)
        Fn_loc_neg_no_thresh = self.get_field_center(self._data['divergence'], rel_th=0.0, abs_th=0.0, sign=-1)
        max_div_idx_img = (Fn_loc_no_thresh * np.array(self.frame_shape) / np.array(self.vf_grid_shape))
        min_div_idx_img = (Fn_loc_neg_no_thresh * np.array(self.frame_shape) / np.array(self.vf_grid_shape))

        dipole_dist_vector = max_div_idx_img - min_div_idx_img
        dipole_unit_vector = dipole_dist_vector / np.linalg.norm(dipole_dist_vector)
        dipole_normal_vector = np.cross(np.append(dipole_unit_vector, 0), [0.0, 0.0, 1.0])[:2]
        dipole_loc = min_div_idx_img + dipole_dist_vector / 2.0

        vector_coords = np.vstack((y,x)).T
        vector_coords = np.reshape(vector_coords, np.append(self.vf_grid_shape, 2))
        vector_coords_dipole_component = np.dot((vector_coords - dipole_loc), dipole_unit_vector)

        # used only for visualization
        flag_vector_positive = vector_coords_dipole_component > 0.0
        vector_coords_positive = vector_coords[flag_vector_positive]
        vector_coords_negative = vector_coords[np.logical_not(flag_vector_positive)]

        # dipole moment calcuation by taking dot product along negative -> positive centroid axis
        dipole_moment_old = np.sum((vector_coords_dipole_component * self._data['divergence'])) / np.prod(self.vf_grid_shape)
        torque_dipole_moment_img_frame_old = dipole_normal_vector * dipole_moment_old
        # flip x and y axis since torque calculations are done in (y,x) coordinates for consistency with image convention
        torque_dipole_moment_old = np.array([torque_dipole_moment_img_frame_old[1], torque_dipole_moment_img_frame_old[0]])

        # dipole moment calculation using integrals
        vector_coords_relative = vector_coords - dipole_loc
        dipole_moment = np.sum((vector_coords_relative * self._data['divergence'][:,:,None]), axis=(0,1)) / np.prod(self.vf_grid_shape)
        torque_dipole_moment_img_frame = np.cross(np.append(dipole_moment, 0), [0.0, 0.0, 1.0])[:2]
        # flip x and y axis since torque calculations are done in (y,x) coordinates for consistency with image convention
        torque_dipole_moment = np.array([torque_dipole_moment_img_frame[1], torque_dipole_moment_img_frame[0]])

        # # temp: check that dipole in normal component is (close to) zero
        # vector_coords_normal_component = np.dot((vector_coords - dipole_loc), dipole_normal_vector)
        # dipole_moment_normal = np.sum((vector_coords_normal_component * self._data['divergence'])) / np.prod(self.vf_grid_shape)
        # print(dipole_moment_normal/dipole_moment)
        # # end temp

        Fx = np.sum(vfield[:,:,0]) / np.prod(self.vf_grid_shape)
        Fy = np.sum(vfield[:,:,1]) / np.prod(self.vf_grid_shape)
        tau_x = torque_dipole_moment[0]
        tau_y = torque_dipole_moment[1]

        tau_x_old = torque_dipole_moment_old[0]
        tau_y_old = torque_dipole_moment_old[1]

        Fz_hhd = np.sum(np.linalg.norm(vd, axis=2)) / np.prod(self.vf_grid_shape)

        if abs(np.max(self._data['curl'])) > abs(np.min(self._data['curl'])):
            rotational_center = c_minus
        else:
            rotational_center = c_plus

        max_curl_dist_vecs = index_mat - rotational_center
        tau_z_hhd = np.sum(np.cross(np.flip(max_curl_dist_vecs, axis=2), vr)) / np.prod(self.vf_grid_shape)

        self._data['Ft'] = np.array([Fx, Fy])
        self._data['Fn'] = Fz_hhd
        self._data['Tn'] = tau_z_hhd
        self._data['Tx'] = np.array([tau_x, tau_y])

        plot_thresh = 0.1
        if abs(Fz_hhd) > plot_thresh:
            Fn_loc_plot = Fn_loc_no_thresh
            Fn_loc_neg_plot = Fn_loc_neg_no_thresh
            dipole_info = [dipole_unit_vector, dipole_normal_vector, dipole_loc, vector_coords_positive, vector_coords_negative]
        else:
            Fn_loc_plot = (0, 0)
            Fn_loc_neg_plot = (0, 0)
            dipole_info = None
        
        if abs(tau_z_hhd) > plot_thresh:
            Tn_loc_plot = rotational_center
        else:
            Tn_loc_plot= (0, 0)

        # Compute grasp shape and angle
        Q, q, angle = self.get_contact_shape(frame, self.ref_frame)
        # Plot
        if display:
            divr_plot = self.plot_field_info(self.surf_plot(self._data['divergence'], 40, 125)*0.6, Fn_loc_plot)
            divr_plot = self.plot_field_info(divr_plot,  Fn_loc_neg_plot, include_vertical=False)
            curl_plot = self.plot_field_info(self.surf_plot(self._data['curl'], 10, 125)*0.6, Tn_loc_plot, include_vertical=tau_z_hhd>0)
            # subtext = f"Ft:       {np.linalg.norm(self._data['Ft']):6.1f}                Fn: {self._data['Fn']:6.1f}                       Tn: {self._data['Tn']:6.1f}"
            subtext = f"Ft:       {np.linalg.norm(self._data['Ft']):6.1f}          Fn: {self._data['Fn']:6.1f}                        " \
                + f"Tx:       {np.linalg.norm(self._data['Tx']):6.1f}          Tn: {self._data['Tn']:6.1f}"
            self.plot_vector_fields((frame*0.6, divr_plot  , curl_plot ),
                                    (vfield*2 , vd*5       , vr*5      ), 
                                    (f'a={np.degrees(angle):.0f}','Divergence', 'Curl'),
                                    # subtext,  self._data['Ft'], self._data['Tx'])
                                    subtext,  self._data['Ft'], self._data['Tx'], 
                                    dipole_info)
        
        # # calibrated torques
        tau_x *= 3.96
        tau_y *= 4.89
        tau_z_hhd *= 4.263

        # publish wrench topic in ros message.
        self.wrench = [Fx, Fy, Fz_hhd, tau_x, tau_y, tau_z_hhd]
        self.publish_wrench(data = self.wrench, publisher = self.pub_tw)

        self.wrench_2 = [Fx, Fy, Fz_hhd, tau_x_old, tau_y_old, tau_z_hhd]
        self.publish_wrench(data = self.wrench_2, publisher = self.pub_tw_2)

        wrench_fingervision = [Fx, Fy, Fz_fingervision, tau_x_fingervision, tau_y_fingervision, tau_z_fingervision]
        self.publish_wrench(data = wrench_fingervision, publisher = self.pub_fg)

        wrench_tactile_hhd = [Fx, Fy, Fz_hhd, 0, 0, tau_z_hhd]
        self.publish_wrench(data = wrench_tactile_hhd, publisher = self.pub_th)

        return [self._data[var] for var in self.out_vars], angle

    """ Plotting """

    def show_view(self):
        """Show preprocessed streamed video from sensor"""
        while True:
            frame = self.get_frame()
            cv2.imshow("Digit View", frame)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        
    def surf_plot(self, matrix, scale, lim):
        """Surface colormap plot of matrix"""
        matrix = matrix*scale + lim
        matrix = np.clip(matrix, 0, 255).astype('uint8')
        if matrix.shape[:2] != self.frame_shape:
            (h, w) = self.frame_shape
            matrix = resize(matrix, (h,w), inter=cv2.INTER_LINEAR)
        return cv2.applyColorMap(matrix, cv2.COLORMAP_JET)

    def vector_field_image(self, frame, vfield, arrowscale=1, brightscale=1):
        """Arrow plot of vector field"""
        frame = (frame*brightscale).astype('uint8')
        return draw_flow(frame, vfield*arrowscale)

    def plot_field_info(self, pic, center, angle=None, arrowlen=50, include_vertical=True):
        """ Write field information """
        p = np.flip(center * np.array(self.frame_shape) / np.array(self.vf_grid_shape))
        pic = draw_point(pic, p, (255,255,255), 10, thickness=1, include_vertical=include_vertical)
        if angle is not None:
            dp = np.array((np.sin(angle), np.cos(angle))) * arrowlen
            pic = draw_vector(pic, p, dp, tipLength=0.1)
        return pic

    # def plot_vector_fields(self, frames, vfields, labels, subtext=None, forcearrow=None, torquearrow=None):
    def plot_vector_fields(self, frames, vfields, labels, subtext=None, forcearrow=None, torquearrow=None, dipole_info=None):
        """Stacked plotting of multiple vector fields and additional information """
        if dipole_info is not None:
            dipole_unit_vector = dipole_info[0]
            dipole_normal_vector = dipole_info[1]
            dipole_loc = dipole_info[2]
            vector_coords_positive = dipole_info[3]
            vector_coords_negative = dipole_info[4]

        vpic = []
        for i, (frame, label, vfield) in enumerate(zip(frames, labels, vfields)):
            frame = frame.astype('uint8')
            pic = self.vector_field_image(frame, vfield)
            if i==0:
                (h, w) = pic.shape[:2]
                cv2.rectangle(pic, (0,0), (w, h), (255,255,255), 4)
            if label: cv2.putText(pic, label, (5,frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1,2)

            if label == "Divergence" and dipole_info is not None:
                # pic = cv2.line(pic, np.flip(dipole_loc - dipole_normal_vector * h).astype("int"), np.flip(dipole_loc + dipole_normal_vector * h).astype("int"), (255,255,255), thickness=1)
                pic = cv2.arrowedLine(pic, tuple(np.flip(dipole_loc).astype("int")), tuple(np.flip(dipole_loc + dipole_unit_vector * 50).astype("int")), (255,0,255), thickness=2)
                pic = cv2.arrowedLine(pic, tuple(np.flip(dipole_loc).astype("int")), tuple(np.flip(dipole_loc + dipole_normal_vector * 50).astype("int")), (255,255,255), thickness=2)
                # for vector_coord_positive in vector_coords_positive:
                #     cv2.circle(pic, np.flip(vector_coord_positive), 2, (0, 0, 255))
                # for vector_coord_negative in vector_coords_negative:
                #     cv2.circle(pic, np.flip(vector_coord_negative), 2, (255, 0, 0))

            vpic.append(pic)
        vpic = np.concatenate(vpic, axis=1)
        t1 = time.time()
        cv2.putText(vpic, f"{1/(t1-self.t):.0f}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),1,2)
        if subtext is not None:
            pic = np.zeros((20, vpic.shape[1], vpic.shape[2]), dtype='uint8')
            cv2.putText(pic, subtext, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1,2)
            vpic = np.concatenate((vpic, pic), axis=0)
            if forcearrow is not None:
                vpic = draw_vector(vpic, (50, vpic.shape[0]-10), forcearrow*4.0)
            if torquearrow is not None:
                vpic = draw_vector(vpic, (500, vpic.shape[0]-10), torquearrow*0.2)

        # imgMsg = self.bridge.cv2_to_imgmsg(vpic, "rgb8")
        imgMsg = self.bridge.cv2_to_imgmsg(vpic, "bgr8")

        # publish vector field image topic in ros.
        self.publish_image(imgMsg, self.pub_vf_img)

        cv2.imshow(f"Vector field", vpic)
        self.t = t1

    def show_vector_field(self):
        """Show processed vector field streamed video"""
        while True:
            self.current_contact_state(display=True)

            # zero image on mouse click
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    finger.get_ref_frame(N=10, save=False, generate_new=True)
            cv2.setMouseCallback("Vector field", mouse_callback)

            if cv2.waitKey(1) == 27: break          
        cv2.destroyAllWindows()
    
    def publish_wrench(self, data, publisher, stamp = None):
		# publish force torque
        wren = WrenchStamped()
        wren.wrench.force.x  = data[0]
        wren.wrench.force.y  = data[1]
        wren.wrench.force.z  = data[2]
        wren.wrench.torque.x = data[3]
        wren.wrench.torque.y = data[4]
        wren.wrench.torque.z = data[5]
        wren.header = Header()

        if stamp is None: 
            wren.header.stamp = rospy.Time.now()
        else:
            wren.header.stamp = stamp

        publisher.publish(wren)
        #rospy.loginfo("data: " + str(np.array(data))) 

    def publish_image(self, imgMsg, pub_img):
		# publish image topic
        pub_img.publish(imgMsg)
        #rospy.loginfo("data: " + str(np.array(data))) 

    # def publish_vector_field(self, vf, stamp = None):
	# 	# publish force torque
    #     vf_msg = vector_field()
    #     vf_msg.header = Header()

    #     if stamp is None: 
    #         vf_msg.header.stamp = rospy.Time.now()
    #     else:
    #         vf_msg.header.stamp = stamp

    #     vf_msg.vector_field = vf

    #     self.pub_vf.publish(vf_msg)


"""(D) AUX FUNCTIONS ------------------------------------------------------------------------"""

class Camera:
    def __init__(self):
        log("[ ]Attempting connection to sensor...")                   
        self.vidcap = cv2.VideoCapture(0)
        while True:
            try:
                log(" • Attempting connection. ")
                if self.vidcap.isOpened():
                    ret, _ = self.vidcap.read()
                    if ret: log(f"[o] Connected to sensor."); break  
                time.sleep(1)
            except KeyboardInterrupt:
                sys.exit()          
    
    def get_frame(self):
        ret, frame = self.vidcap.read()
        frame = resize(frame, (320,240), inter=cv2.INTER_AREA) 
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def disconnect(self):
        pass

""" Computations """
def compute_channel_flow(f0,f1):
    return cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 3, 32, 3, 5, 1.2, 0)

def compute_optical_flow(f0, f1, channels=np.arange(3)):
    """ Compute optical flow channelwise and average """
    if not hasattr(channels, '__len__'): channels = (channels,)
    shape = f1.shape[:2]+(2,)
    f0 = [cv2.UMat(f0[...,i]) for i in channels]
    f1 = [cv2.UMat(f1[...,i]) for i in channels]
    flow = cv2.UMat(np.zeros(shape, dtype='float32'))
    for frames in zip(f0,f1):
        flow_channel = compute_channel_flow(*frames)
        flow=cv2.add(flow,flow_channel)
    return flow.get()/len(channels)

def get_curl(vfield):
    kerndx=np.array([[0,0,0],[1, 0, -1],[0,0,0]])
    kerndy=np.array([[0,1,0],[0, 0, 0],[0,-1,0]])
    dudy=signal.convolve(vfield[...,0],kerndy,mode='same')
    dvdx=signal.convolve(vfield[...,1],kerndx,mode='same')
    return dvdx-dudy

def get_center_of_mass(a):
    a = a.astype('float32')
    M10 = np.sum(a.T*np.arange(a.shape[0]).T)
    M01 = np.sum(a*np.arange(a.shape[1]))
    M00 = max(1e-5, np.sum(a))
    return np.array((M10/M00, M01/M00))

def get_orientation(a, p):
    a = a.astype('float32')
    xx = np.arange(a.shape[0]).T - p[0]
    yy = np.arange(a.shape[1])   - p[1]
    mu20 = np.sum(a.T * xx**2)
    mu02 = np.sum(a   * yy**2)
    mu11 = np.sum((a.T * xx).T*yy)
    return 0.5*np.arctan2(2*mu11,(mu20-mu02))

""" Plotting """
def draw_flow(img, flow, step=16, arrow_size=1):
    """ Draw vector field of the optical flow """
    img1 = img.copy()
    h, w = img.shape[:2]
    hf, wf = flow.shape[:2]
    red_size_vf = img.shape[:2]!=flow.shape[:2]
    (sh, sw) = (int(h/hf), int(w/wf)) if red_size_vf else (step, step)
    y, x = np.mgrid[sh/2:h:sh, sw/2:w:sw].reshape(2,-1).astype(int)
    fx, fy = flow.reshape(-1,2).T if red_size_vf else flow[y,x].T 
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img1, lines, 0, (255, 255, 255), arrow_size)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img1, (x1, y1), int(arrow_size*1.5), (255, 255, 255), -1)
    return img1

def draw_hsv(flow):
    """ Draw hsv representation of the optical flow """
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*10, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def draw_vector(frame, p0, dp, color=(255, 255, 255), thickness=1, tipLength=0.4):
    """ Draw equivalent (averaged) vector of the vector field """
    p0 = (np.array(p0)).astype(int)
    dp = np.array(dp).astype(int)
    p1 = p0+dp
    image = cv2.arrowedLine(frame, tuple(p0), tuple(p1), color, thickness, tipLength=tipLength)
    return frame

def draw_point(img, pos, color, radius, thickness=2, include_vertical=True):
        lv = (np.array(((0,-radius),(0,radius))) + np.array(pos)).astype('int')
        lh = (np.array(((-radius,0),(radius,0))) + np.array(pos)).astype('int')
        if include_vertical:
            img = cv2.line(img, tuple(lv[0]), tuple(lv[1]), color, thickness=thickness)
        img = cv2.line(img, tuple(lh[0]), tuple(lh[1]), color, thickness=thickness)
        return img

def get_divergence(vfield):
    kerndx=np.array([[0,0,0],[1, 0, -1],[0,0,0]])
    kerndy=np.array([[0,1,0],[0, 0, 0],[0,-1,0]])
    dudx=signal.convolve(vfield[...,0],kerndx,mode='same')
    dvdy=signal.convolve(vfield[...,1],kerndy,mode='same')
    return dudx+dvdy

def setup_OpenCL():
    log("OpenCV compute status", format=('purple','bold'))
    cv2.ocl.setUseOpenCL(True)
    log(
        f'- OpenCL available: {cv2.ocl.haveOpenCL()}\n'+\
        f'- OpenCL active:    {cv2.ocl.useOpenCL()}\n',
        format=('purple','italic')
        )

""" (E) GRASP ANGLE ESTIMATION -------------------------------------------------------------- """

class ContactShape:
    """ Based on PyTouch's class of the same name: https://github.com/facebookresearch/PyTouch.
    Fits an elipse to the estimated shape of the contact, then computes it's angle """
    def __init__(
        self, draw_poly=True, contour_threshold=90, *args, **kwargs
    ):
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold
        self.angle = 0
        # ADJUSTABLE PARAMETERS ================================
        self.angle_filter = 0.95 # filter for angle estimation (noise removal)
        self.diff_thresh = 0.13 # Thresholding applied on the difference between frame and ref_frame
        self.diff_scale = 1.2 # Scaling factor applied on the difference between frame and ref_frame
        self.ch_weights = np.array((1,1.2,0.8)) # weights of each color of RGB on the difference computation
        self.mask_thresh = 0.072 # Thresholding for contour detection over a binary mask
        self.blur_kernel = 64 # blurring image for broader contour detection
        # ======================================================

    def __call__(self, target, base):
        #event_pic = self._callibrate(ref_frames, target)
        self.shape = target.shape
        diff = self._diff(target*self.ch_weights, base*self.ch_weights)
        # cv2.imshow(f"diff", diff) #cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)[1])
        diff = self._smooth(diff)
        contours = self._contours(diff)
        major_axis, major_axis_end, minor_axis, minor_axis_end, angle = (0,0),(0,0),(0,0),(0,0), 0
        if contours:
            (
                poly,
                major_axis,
                major_axis_end,
                minor_axis,
                minor_axis_end,
            ) = self._compute_contact_area(contours, self.contour_threshold)
            if self.draw_poly and poly is not None:
                self._draw_major_minor(
                    target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
                )
            angle = self._compute_contact_angle(major_axis, major_axis_end)
        return (major_axis, major_axis_end), (minor_axis, minor_axis_end), angle

    def _diff(self, target, base):
        diff = np.abs(target - base) / 255.0
        diff[diff < self.diff_thresh] = 0
        diff_abs = np.mean(diff, axis=-1) * self.diff_scale
        return diff_abs
        
    def _smooth(self, target):
        kernel = np.ones((self.blur_kernel, self.blur_kernel), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        return diff_blur

    def _contours(self, target):
        mask = cv2.UMat(((np.abs(target) > self.mask_thresh) * 255).astype(np.uint8))
        kernel = cv2.UMat(np.ones((16, 16), np.uint8))
        mask = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_major_minor(
        self,
        target,
        poly,
        major_axis,
        major_axis_end,
        minor_axis,
        minor_axis_end,
        lineThickness=2,
    ):
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])),
            (int(major_axis[0]), int(major_axis[1])),
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        for contour in contours:
            if len(contour.get()) > contour_threshold:
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
            else:
                poly = None 
                major_axis, major_axis_end, minor_axis, minor_axis_end = (0,0),(0,0),(0,0),(0,0)
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end

    def _compute_contact_angle(self, major_axis, major_axis_end):
        x = major_axis_end[0]-major_axis[0]
        y = major_axis_end[1]-major_axis[1]
        a = -np.arctan2(abs(x), abs(y)) * np.sign(y)
        self.angle = self.angle*self.angle_filter + a*(1-self.angle_filter)
        return self.angle


 
"""======================================================================================"""

if __name__ == '__main__':
    setup_OpenCL()

    rospy.init_node('touch_sensor', anonymous=True)

    finger = TouchSensor(sensor='GS_MINI', equalizer='none')
    # finger = TouchSensor(sensor='DIGIT', equalizer='none')
    # finger.show_view()
    finger.get_ref_frame(N=10, save=False, generate_new=True)
    finger.show_vector_field()
    # finger.current_contact_state(display=True)
    # cv2.waitKey(2)
