# Runs only when ros is sourced .i.e., run this file with python2 only after doing 'source /opt/ros/kinetic/setup.bash'
# As this needs python2.7 with ros python packages
import cv2
from rosbag import bag
import cv_bridge

def show_and_save_images(iterator, prefix, show=False):
	bridge = cv_bridge.CvBridge()

	count = 0

	for idx, (topic, msg, tim) in enumerate(iterator):
		# if (idx%2 == 0):
		# if True:
		# count = count + 1
		im_name = prefix + "_" + str(idx)
		cv_im = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
		if show:
			cv2.namedWindow(im_name, cv2.WINDOW_FREERATIO)
			cv2.imshow(im_name, cv_im)
			cv2.waitKey()
			cv2.destroyAllWindows()
		cv2.imwrite(im_name + ".png", cv_im)
		# elif count >= 60:
			# break

# Read bag file
bag_file = bag.Bag("with_depth/depth.bag")

# Create image and mask iterators
im_iter = bag_file.read_messages(['/image'])
mask_iter = bag_file.read_messages(['/mask'])
depth_iter = bag_file.read_messages(['/depth'])

show_and_save_images(im_iter, "frames_rgb/c1_image", show=False)
show_and_save_images(depth_iter, "frames/c1_depth", show=False)
show_and_save_images(mask_iter, "frames/c1_mask", show=False)

