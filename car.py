"""
Author: YWiyogo
"""
import numpy as np


class Car:
    """Car Class for tracking"""

    def __init__(self, carid, bbox):
        """Constructor"""
        self.carID = carid
        self.centroid = self.calc_centroid(bbox)
        self.bbox = bbox
        self.tracked_count = 0
        self.frame_update = True
        self.noupdate_count = 0

    def update_state(self, new_bbox):
        """Updating car state. This function can be call max. once/frame"""
        centroid = self.calc_centroid(new_bbox)
        if not self.frame_update:
            self.noupdate_count = self.noupdate_count - 1
            self.frame_update = True
            if self.check_centroid(centroid):
                # avg_x_top = int(np.average([self.bbox[0][0], new_bbox[0][0]]))
                # avg_y_top = int(np.average([self.bbox[0][1], new_bbox[0][1]]))
                # avg_x_bottom = int(np.average([self.bbox[1][0], new_bbox[1][0]]))
                # avg_y_bottom = int(np.average([self.bbox[1][1], new_bbox[1][1]]))
                # self.bbox = ((avg_x_top, avg_y_top), (avg_x_bottom, avg_y_bottom))
                self.bbox = new_bbox
                self.tracked_count = self.tracked_count + 1
            return True
        else:
            print("Update failed of car -> restart counter", self.carID)
            self.tracked_count = 0
            self.noupdate_count = 0
            return False

    def check_bbox(self, new_bbox):
        """Check the validity of the bounding box for this car object"""
        centroid = self.calc_centroid(new_bbox)
        return self.check_centroid(centroid)

    def check_centroid(self, centroid):
        """Check the validity of the validity of the centroid to this object"""
        if centroid[0] > self.bbox[0][0] and centroid[0] < self.bbox[1][0] and centroid[1] > self.bbox[0][1] and centroid[1] < self.bbox[1][1]:
            return True
        else:
            diff_x = abs(self.centroid[0] - centroid[0])
            diff_y = abs(self.centroid[1] - centroid[1])
            distance = np.sqrt(diff_x**2 + diff_y**2)
            thres = 60
            return True if distance < thres else False

    def calc_centroid(self, bbox):
        """Calculate the centroid given a bounding box"""
        return (int(np.average([bbox[0][0], bbox[1][0]])),
                int(np.average([bbox[0][1], bbox[1][1]])))

    def is_valid(self):
        """
        Check if the existence of this car can be eliminated or not.
        Call this function an the end of the pipeline
        """
        if self.frame_update is False:
            self.noupdate_count = self.noupdate_count + 1
            self.tracked_count = self.tracked_count - 1
            if self.noupdate_count > -1:
                return False
            else:
                return True
        else:
            return True
