import os
from configparser import ConfigParser

import click
import pandas as pd

import simba.labelling_aggression as la
from loguru import logger
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

    def merge_with_template(self, template):
        if not os.path.exists(template):
            logger.error("Template {} does not exist.".format(template))

        # refresh myself before the merge
        self.read_from_disk()

        t = ConfigParser()
        t.read(template)

        # all sections should be imported
        for s in t.sections():
            if s not in self.config.sections():
                self.config.add_section(s)

        # enumerate template key-val and import values that are not
        # found in me.
        for s in t.sections():
            for o in t.options(s):
                val_in_template = t.get(s, o)
                val_in_me = self.config.get(s, o, fallback=None)

                # we import template value if I don't have one.
                if not val_in_me:
                    self.config.set(s, o, val_in_template)

                # if mine is 0, use the template's
                if val_in_me == "0":
                    self.config.set(s, o, val_in_template)

                # if mine is "no", use template's
                if val_in_me == "no":
                    self.config.set(s, o, val_in_template)

        # write back to disk
        self.write_to_disk()

    def set_distance_mm(self, distance):
        self.config.set("Frame settings", "distance_mm", str(distance))

    def set_mm_per_pixel(self, ppm):
        self.config.set("Frame settings", "mm_per_pixel", str(ppm))

    def set_roi_settings_no_of_animals(self, no):
        self.config.set("ROI settings", "no_of_animals", str(no))

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
    "--animal-no",
    "-n",
    required=True,
    default=2,
    help="No of animals to analyze.",
)
@click.option(
    "--video",
    "-v",
    required=True,
    help="Path to video. The video will be copied to project folder.",
)
@click.option(
    "--csv", "-s", required=True, help="Path to video's Deeplabcut CSV."
)
@click.option(
    "--template-ini",
    "-t",
    required=False,
    help="Template config ini used to import analysis settings.",
)
@click.argument("name", nargs=1)
def create(
    path,
    classifiers,
    tracking_method,
    body_parts,
    animal_no,
    video,
    csv,
    name,
    template_ini=None,
):

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

    if template_ini:
        config = MyConfig(path_to_ini)
        config.merge_with_template(template_ini)


cli.add_command(create)


@click.command()
@click.option(
    "--classifier", required=True, help="Classifier you want to compute w/."
)
@click.option("--skip-plots", default=0, help="True to skip visualizations.")
@click.option(
    "--skip-labelling", default=0, help="True to skip agression labelling."
)
@click.option(
    "--skip-model-validation", default=0, help="True to skip model validation."
)
@click.option("--skip-training", default=0, help="True to skip model training.")
@click.option(
    "--skip-modelling", default=0, help="True to skip running models entirely."
)
@click.argument("path-to-ini", nargs=1)
def analyze(
    path_to_ini,
    classifier,
    skip_plots,
    skip_labelling,
    skip_model_validation,
    skip_training,
    skip_modelling,
):
    current_video = "tmp"

    config = MyConfig(path_to_ini)

    # extract frames
    logger.debug("Extract video frames")
    videopath = os.path.join(os.path.dirname(path_to_ini), "videos")
    extract_frames_ini(videopath, path_to_ini)

    # generate video parameters
    logger.debug("Generate video info CSV")
    v = video_info_table(path_to_ini)
    v.generate_video_info_csv()
    # TODO: video SAV's PPM will be default to 0!
    ppm = config.config.getfloat("Frame settings", "mm_per_pixel")
    logger.debug("-" * 50)
    logger.debug(ppm)
    project_path = config.config.get("General settings", "project_path")
    video_info_csv = os.path.join(project_path, "logs", "video_info.csv")
    df = pd.read_csv(video_info_csv, dtype={"pixels/mm": float})
    df.at[0, "pixels/mm"] = ppm
    df.to_csv(video_info_csv, index=False)

    # create outlier settings
    logger.debug("Correct outliers")
    o = outlier_settings(path_to_ini)

    # correct outlier
    l = loadprojectini(path_to_ini)
    l.correct_outlier()

    # extract features
    # TODO: output feature csv path is coded, and is not returned!
    # This makes it impossible to automate `trainmodel` w/o also
    # hardcoding the path. This is a pitfall of simba's design.
    logger.debug("Extract features")
    l.extractfeatures()
    extracted_features_csv = os.path.join(
        config.config.get("General settings", "csv_path"),
        "features_extracted",
        "{}.csv".format(current_video),
    )

    # label behavior
    if not skip_labelling:
        logger.debug("Simulation of labelling behaviors")
        frame_path = os.path.join(config.get_frame_input_path(), current_video)
        la.frames_in = [x for x in os.listdir(frame_path) if ".png" in x]
        la.reset()
        la.frames_in = sorted(la.frames_in, key=lambda x: int(x.split(".")[0]))

        # initialize module globals
        number_of_targets = config.config.get("SML settings", "No_targets")
        for i in range(1, int(number_of_targets) + 1):
            target = config.config.get("SML settings", "target_name_" + str(i))
            la.columns.append(target)
            la.behaviors.append(0)
            la.df[la.columns[i - 1]] = 0

        la.current_video = current_video
        la.curent_frame_number = 0
        la.behaviors[0] = 1
        la.save_values(0, 50)
        la.behaviors[0] = 0
        la.save_values(51, len(la.frames_in))

        output_file = os.path.join(
            config.config.get("create ensemble settings", "data_folder"),
            "{}.csv".format(current_video),
        )
        data = pd.read_csv(extracted_features_csv)
        new_data = pd.concat([data, la.df], axis=1)
        new_data = new_data.fillna(0)
        new_data.rename(columns={"Unnamed: 0": "scorer"}, inplace=True)
        new_data.to_csv(output_file, index=False)

    # train single model
    if not skip_training:
        logger.debug("Training model")
        trainmodel2(path_to_ini)

        # default value for this setting is wrong.
        target = config.config.get("SML settings", "target_name_1")
        model_path = config.config.get("SML settings", "model_dir")
        generated_model_path = os.path.join(
            model_path, "generated_models", "{}.sav".format(target)
        )
        config.config.set("SML settings", "model_path_1", generated_model_path)
        config.write_to_disk()

        # proceed
        validate_model_one_vid_1stStep(
            path_to_ini, extracted_features_csv, generated_model_path
        )

    # validate model single vid
    if not skip_model_validation:
        logger.debug("Validate video model")
        discrimination_threshold = config.config.get(
            "threshold_settings", "threshold_1"
        )
        min_bout_length = config.config.get(
            "Minimum_bout_lengths", "min_bout_1"
        )
        generate_gannt = (
            config.config.get("validation/run model", "save_gantt") == "yes"
        )
        # generate_gannt = 0  # 1 to generate
        validate_model_one_vid(
            path_to_ini,
            extracted_features_csv,
            generated_model_path,
            discrimination_threshold,
            min_bout_length,
            generate_gannt,
        )

    # run RF model
    if not skip_modelling:
        logger.debug("Run RF model")
        rfmodel(path_to_ini)

    # analyze machine prediction
    logger.debug("Analyze machine prediction")
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
    logger.debug("Analyze distance/velocity")

    # analyze distances/velocity
    ROI_process_movement(path_to_ini)

    # time bins machine prediction
    logger.debug("Analyze w/ time bin")
    time_bin_size = config.config.getint(
        "Analysis settings", "distance_velocity_time_bin_size"
    )
    time_bins_classifier(path_to_ini, time_bin_size)

    # time bins distance/velocity
    time_bins_movement(path_to_ini, time_bin_size)

    # analyze severity
    logger.debug("Analyze severity")
    severity_scale = config.config.getint("Analysis settings", "severity_scale")
    analyze_process_severity(path_to_ini, severity_scale, classifier)

    if skip_plots:
        return

    # plot gannt
    logger.debug("Plotting gannt chart")
    ganntplot_config(path_to_ini)

    # plot sklearn result
    logger.debug("Plotting sklearn results")
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
    body_part = config.config.get("Heatmap settings", "body_part")
    bin_size_pixels = config.config.getint(
        "Heatmap settings", "bin_size_pixels"
    )
    scale_max_seconds = config.config.get(
        "Heatmap settings", "scale_max_seconds"
    )

    # values: gnuplot2, plasma, magma, jet, inferno, viridis
    palette = config.config.get("Heatmap settings", "palette")
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
