# Runs only when ros is sourced .i.e., run this file with python2 only after doing 'source /opt/ros/kinetic/setup.bash'
# As this needs python2.7 with ros python packages
import csv
from rosbag import bag

def ts_syncer(gt_ts, bag_ts):
	bts_itr = iter(bag_ts)
	bt = next(bts_itr)

	prev_gts = gt_ts[0]
	for gts in gt_ts:
		if gts > bt:
			yield bt, prev_gts
			bt = next(bts_itr)
		prev_gts = gts

# Read timestamps from bag file
bag_file = bag.Bag("minihattan.bag")
image_topic_iter = bag_file.read_messages(['/image'])
bag_ts = [float("%.6f" % ts.to_sec()) for topic, msg, ts in image_topic_iter]

# Read timestamps from ground truth file
gt_file = open('groundtruth.txt', 'r')
reader = csv.reader(gt_file, delimiter='\t')
gt_ts = [float(row[0][:10] + '.' + row[0][10:16]) for row in reader]

# Sync and write time stamps to file
sync_file = open('timestamp_sync.txt', 'w')
for bt, gt in ts_syncer(gt_ts, bag_ts):
	sync_file.write("%.6f\t%.6f\n" % (bt, gt))
sync_file.close()
