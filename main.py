import os
from configparser import ConfigParser

import click
import pandas as pd

from simba.process_data_log import analyze_process_data_log
from simba.process_severity import analyze_process_severity
from simba.ROI_add_to_features import ROItoFeatures
from simba.ROI_analysis_2 import roiAnalysis
from simba.ROI_process_movement import ROI_process_movement
from simba.run_RF_model import rfmodel
from simba.runmodel_1st import validate_model_one_vid_1stStep
from simba.SimBA import loadprojectini
from simba.SimBA import outlier_settings
from simba.SimBA import trainmachinemodel_settings
from simba.SimBA import video_info_table
from simba.timeBins_classifiers import time_bins_classifier
from simba.timeBins_movement import time_bins_movement
from simba.train_model_2 import trainmodel2
from simba.validate_model_on_single_video import validate_model_one_vid


class MyConfig:

    def __init__(self, path_to_ini):
        self.path_to_ini = path_to_ini
        self.config = ConfigParser()
        self.read_from_disk()

    def read_from_disk(self):
        self.config.read(self.path_to_ini)

    def write_to_disk(self):
        with open(self.path_to_ini, "w") as f:
            self.config.write(f)

    def set_distance_mm(self, distance):
        self.config.set("Frame settings", "distance_mm", str(distance))

    def set_mm_per_pixel(self, ppm):
        self.config.set("Frame settings", "mm_per_pixel", str(ppm))

    def set_outlier_movement_criterion(self, criterion):
        self.config.set("Outlier settings",
                        "movement_criterion", str(criterion))

    def set_outlier_location_criterion(self, criterion):
        self.config.set("Outlier settings",
                        "location_criterion", str(criterion))

    def set_outlier_mean_or_median(self, what="mean"):
        self.config.set("Outlier settings", "mean_or_median", str(what))

    def set_process_movements(self, body_parts):
        for index, bp in enumerate(body_parts):
            self.config.set(
                "process movements", "animal_" + str(index + 1) + "_bp", bp
            )
        self.config.set("process movements", "no_of_animals",
                        str(len(body_parts)))

    def set_roi_settings_no_of_animals(self, no):
        self.config.set("ROI settings", "no_of_animals", str(no))

    def set_heat_locations(self, body_part, palette, classifier, bin_size, scale):
        self.config.set("Heatmap location", "body_part", body_part)
        self.config.set("Heatmap location", "Palette", palette)
        self.config.set(
            "Heatmap location",
            "Scale_max_seconds",
            str(scale))
        self.config.set(
            "Heatmap location",
            "bin_size_pixels", str(bin_size))

    def get_generated_model_path(self, classifier):
        return self.config.get("SML settings", "model_path_1")

    def get_discrimination_threshold(self):
        return self.config.get("threshold_settings", "threshold_1")

    def get_min_bout_length(self):
        return self.config.get("Minimum_bout_lengths", "min_bout_1")

@click.group()
def cli():
    pass


@click.command()
@click.option("--classifier",
              required=True,
              help="Classifier you want to compute w/.")
@click.argument("path-to-ini", nargs=1)
def config(path_to_ini, classifier):
    config = MyConfig(path_to_ini)

    # set ppm
    config.set_distance_mm(245)
    config.set_mm_per_pixel(9.124)
    config.write_to_disk()

    # generate video parameters
    v = video_info_table(path_to_ini)
    v.generate_video_info_csv()

    # create outlier settings
    o = outlier_settings(path_to_ini)
    o.set_outliersettings()
    config.read_from_disk()
    config.set_outlier_movement_criterion(0.7)
    config.set_outlier_location_criterion(1.5)
    config.set_outlier_mean_or_median("mean")
    config.write_to_disk()

    # correct outlier
    l = loadprojectini(path_to_ini)
    l.correct_outlier()

    # extract features
    l.extractfeatures()

    # label behavior
    # TBD

    # load metadata
    t = trainmachinemodel_settings(path_to_ini)

    # TODO: path to meta is hardcoded
    # t.load_choosedata.filePath.set(
    #    "/app/output/feng/models/generated_models/mating_meta.csv")
    # t.load_RFvalues()

    # train single model
    # trainmodel2(path_to_ini)

    # run model
    csv = "/app/output/feng/project_folder/csv/features_extracted/tmp.csv"
    config.read_from_disk()
    sav = config.get_generated_model_path("mating")
    validate_model_one_vid_1stStep(path_to_ini, csv, sav)

    # validate model single vid
    discrimination_threshold = config.get_discrimination_threshold()
    min_bout_length = config.get_min_bout_length()
    generated_gannt = 0  # 1 to generate
    # validate_model_one_vid(
    #    path_to_ini, csv, sav, discrimination_threshold, min_bout_length, 0
    #)

    # run RF model
    rfmodel(path_to_ini)

    # analyze machine prediction
    cols = [
        "# bout events",
        "total events duration (s)",
        "mean bout duration (s)",
        "median bout duration (s)",
        "first occurance (s)",
        "mean interval (s)",
        "median interval (s)",
    ]
    analyze_process_data_log(path_to_ini, cols)

    # analyze distance/velocity
    animal_vars = ["Ear_left_1", "Ear_left_2"]
    config.set_process_movements(animal_vars)

    body_part = "Center_1"
    bin_size_pixels = 200
    scale_max_seconds = "auto"
    palette = "gnuplot2"
    config.set_heat_locations(
        body_part, palette, classifier,
        bin_size_pixels, scale_max_seconds)

    config.set_roi_settings_no_of_animals(2)
    config.write_to_disk()

    # analyze distances/velocity
    ROI_process_movement(path_to_ini)

    # time bins machine prediction
    time_bin_size  = 20
    time_bins_classifier(path_to_ini, time_bin_size)

    # time bins distance/velocity
    time_bins_movement(path_to_ini, time_bin_size)

    # analyze severity
    severity_scale = 5 # must > 1!
    analyze_process_severity(path_to_ini, severity_scale, classifier)

    #roiAnalysis(path_to_ini, "features_extracted")

    #ROItoFeatures(path_to_ini)


cli.add_command(config)

if __name__ == "__main__":
    cli()
