import os
from configparser import ConfigParser

import click
import pandas as pd

import simba.labelling_aggression as la
from simba.create_project_ini import write_inifile
from simba.data_plot import data_plot_config
from simba.gantt import ganntplot_config
from simba.import_videos_csv_project_ini import copy_singlevideo_ini
from simba.import_videos_csv_project_ini import extract_frames_ini
from simba.line_plot import line_plot_config
from simba.merge_frames_movie import mergeframesPlot
from simba.path_plot import path_plot_config
from simba.plot_heatmap import plotHeatMap
from simba.plot_threshold import plot_threshold
from simba.process_data_log import analyze_process_data_log
from simba.process_severity import analyze_process_severity
from simba.ROI_add_to_features import ROItoFeatures
from simba.ROI_analysis_2 import roiAnalysis
from simba.ROI_process_movement import ROI_process_movement
from simba.run_RF_model import rfmodel
from simba.runmodel_1st import validate_model_one_vid_1stStep
from simba.SimBA import loadprojectini
from simba.SimBA import outlier_settings
from simba.SimBA import project_config
from simba.SimBA import video_info_table
from simba.sklearn_plot_scripts.plot_sklearn_results_2 import plotsklearnresult
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
            self.config.set("process movements", "animal_" +
                            str(index + 1) + "_bp", bp)
        self.config.set("process movements", "no_of_animals",
                        str(len(body_parts)))

    def set_roi_settings_no_of_animals(self, no):
        self.config.set("ROI settings", "no_of_animals", str(no))

    def set_heat_locations(self, body_part, palette, classifier, bin_size, scale):
        self.config.set("Heatmap location", "body_part", body_part)
        self.config.set("Heatmap location", "Palette", palette)
        self.config.set("Heatmap location", "Scale_max_seconds", str(scale))
        self.config.set("Heatmap location", "bin_size_pixels", str(bin_size))

    def set_classifier(self, classifier):
        self.config.set("create ensemble settings", "classifier", classifier)

    def get_generated_model_path(self, classifier):
        return self.config.get("SML settings", "model_path_1")

    def get_discrimination_threshold(self):
        return self.config.get("threshold_settings", "threshold_1")

    def get_min_bout_length(self):
        return self.config.get("Minimum_bout_lengths", "min_bout_1")

    def get_frame_input_path(self):
        return self.config.get("Frame settings", "frames_dir_in")


@click.group()
def cli():
    pass


@click.command()
@click.option("--path", "-p", required=True, help="Path to project files.")
@click.option(
    "--classifiers",
    "-c",
    required=True,
    help="CSV w/o space in between for multiple classifiers.",
)
@click.option(
    "--tracking-method",
    "-t",
    default="classic",
    help='Tracking method, ["classic", "multi"].',
)
@click.option(
    "--body-parts",
    "-b",
    required=True,
    default=16,
    help="How many body parts to track.",
)
@click.option(
    "--animal-no", "-n", required=True, default=2, help="No of animals to analyze."
)
@click.option(
    "--video",
    "-v",
    required=True,
    help="Path to video. The video will be copied to project folder.",
)
@click.option("--csv", "-s", required=True, help="Path to video's Deeplabcut CSV.")
@click.argument("name", nargs=1)
def create(path, classifiers, tracking_method, body_parts, animal_no, video, csv, name):
    targets = classifiers.split(",")

    # compute index to pull body part header from  pose config
    BP_TO_INDEX_MAPPING = {
        "classic": {
            (1, 4): 0,
            (1, 7): 1,
            (1, 8): 2,
            (1, 9): 3,
            (2, 8): 4,
            (2, 14): 5,
            (2, 16): 6,
            (0, 987): 7,
            (0, "user_defined"): None,
        },
        "multi": {8: 9, 14: 10, 16: 11, "user_defined": None},
    }
    mapping = BP_TO_INDEX_MAPPING.get(tracking_method, None)
    if mapping is None:
        print(
            "Can not determine pose mapping. Tracking method {} is not defined.".format(
                tracking_method
            )
        )
        return
    if tracking_method == "classic":
        pose_index = mapping.get((animal_no, body_parts), None)
        if pose_index is None:
            print("Can not determine pose index. Quit.")
            return
    if tracking_method == "multi":
        pose_index = mapping.get(body_parts, None)
        if pose_index is None:
            print("Can not determine pose index. Quit.")
            return

    path_to_ini = write_inifile(
        msconfig="yes",  # don't know what this is, hardcoded in original.
        project_path=path,
        project_name=name,
        no_targets=len(targets),
        target_list=targets,
        bp=body_parts,
        listindex=pose_index,
        animalNo=str(animal_no),
        csvorparquet="csv",  # workflow type
    )

    # importing single video
    copy_singlevideo_ini(path_to_ini, video)

    # importing csv
    # Note: Unfortunately, there are data cleanse happening in Simba code,
    # so I have to use its code instead of a direct func call.
    config = project_config()
    config.configinifile = path_to_ini
    config.file_csv.filePath.set(csv)
    config.import_singlecsv()


cli.add_command(create)


@click.command()
@click.option("--classifier", required=True, help="Classifier you want to compute w/.")
@click.option("--skip-plots", default=1, help="True to skip visualizations.")
@click.option("--skip-labelling", default=1, help="True to skip agression labelling.")
@click.option(
    "--skip-video-validation", default=1, help="True to skip video validation."
)
@click.argument("path-to-ini", nargs=1)
def analyze(path_to_ini, classifier, skip_plots, skip_labelling, skip_video_validation):
    current_video = "tmp"

    config = MyConfig(path_to_ini)
    config.set_classifier(classifier)

    # set ppm
    config.set_distance_mm(245)
    config.set_mm_per_pixel(9.124)
    config.write_to_disk()

    # extract frames
    videopath = os.path.join(os.path.dirname(path_to_ini), "videos")
    extract_frames_ini(videopath, path_to_ini)

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
    if not skip_labelling:
        frame_path = os.path.join(config.get_frame_input_path(), current_video)
        la.frames_in = [x for x in os.listdir(frame_path) if ".png" in x]
        la.reset()
        la.frames_in = sorted(la.frames_in, key=lambda x: int(x.split(".")[0]))

        la.current_video = current_video
        la.configure(path_to_ini)

        # la.choose_folder(path_to_ini)
        la.curent_frame_number = 0
        la.behaviors[0] = 1
        la.save_values(0, 50)
        la.behaviors[0] = 0
        la.save_values(51, len(la.frames_in))

        input_file = os.path.join(
            config.config.get("General settings", "csv_path"),
            "features_extracted",
            "{}.csv".format(current_video),
        )
        output_file = os.path.join(
            config.config.get("create ensemble settings", "data_folder"),
            "{}.csv".format(current_video),
        )
        data = pd.read_csv(input_file)
        new_data = pd.concat([data, la.df], axis=1)
        new_data = new_data.fillna(0)
        new_data.rename(columns={"Unnamed: 0": "scorer"}, inplace=True)

        new_data.to_csv(output_file, index=False)
        print(output_file)
        print("Annotation file for {} created.".format(current_video))

    # train single model
    trainmodel2(path_to_ini)

    # run model
    csv = "/app/output/testme2/project_folder/csv/features_extracted/tmp.csv"
    config.read_from_disk()
    sav = config.get_generated_model_path(classifier)
    validate_model_one_vid_1stStep(path_to_ini, csv, sav)

    # validate model single vid
    discrimination_threshold = config.get_discrimination_threshold()
    min_bout_length = config.get_min_bout_length()
    generate_gannt = 0  # 1 to generate
    if not skip_video_validation:
        validate_model_one_vid(
            path_to_ini,
            csv,
            sav,
            discrimination_threshold,
            min_bout_length,
            generate_gannt,
        )

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

    # values: gnuplot2, plasma, magma, jet, inferno, viridis
    palette = "gnuplot2"
    config.set_heat_locations(
        body_part, palette, classifier, bin_size_pixels, scale_max_seconds
    )

    config.set_roi_settings_no_of_animals(2)
    config.write_to_disk()

    # analyze distances/velocity
    ROI_process_movement(path_to_ini)

    # time bins machine prediction
    time_bin_size = 20
    time_bins_classifier(path_to_ini, time_bin_size)

    # time bins distance/velocity
    time_bins_movement(path_to_ini, time_bin_size)

    # analyze severity
    severity_scale = 5  # must > 1!
    analyze_process_severity(path_to_ini, severity_scale, classifier)

    if skip_plots:
        return

    # plot gannt
    ganntplot_config(path_to_ini)

    # plot sklearn result
    create_video = 1  # 0 or 1
    create_frame = 1  # 0 or 1
    plotsklearnresult(path_to_ini, create_video, create_frame)

    # data plot
    data_plot_config(path_to_ini, "Centroid")

    # path plot
    path_plot_config(path_to_ini)

    # distance plot
    line_plot_config(path_to_ini)

    # heatmap
    body_part = "Center_1"
    last_image_only = 0  # 0 or 1
    plotHeatMap(
        path_to_ini,
        body_part,
        bin_size_pixels,
        scale_max_seconds,
        palette,
        classifier,
        last_image_only,
    )

    # plot threshold
    plot_threshold(path_to_ini, classifier)

    # merge frames
    merging_these = [
        ("Sklearn", 1),
        ("Gantt", 1),
        ("Path", 1),
        ("'Live' data", 1),
        ("Distance", 1),
        ("Probability", 1),
    ]
    try:
        mergeframesPlot(path_to_ini, [x[1] for x in merging_these])
    except UnboundLocalError:
        print("Known bug. Ignore.")

    return

    # roiAnalysis(path_to_ini, "features_extracted")

    # ROItoFeatures(path_to_ini)


cli.add_command(analyze)

if __name__ == "__main__":
    cli()
