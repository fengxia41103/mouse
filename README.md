# Dev & deployment

Pre-requisite: Linux, because simba is a GUI app, and would require X window feature to use the host's display.

Two dockers will be created: Simba is based on a `miniconda` base image, and `deeplabcut` on Python:3.7.

1. On Linux host, install docker & docker-compose, eg. `pip install docker-compose`
2. Open a terminal and run `xhost +`. This allows display to be
   x-window-ed from docker to your host.
3. For the first run: `docker-compose up --build -d`. This will take a
   while as it's pulling a lot of things from conda repo.


For consequetive runs, `docker-compose up -d simba` if you only want
simba to run. For both deeplabcut & simba, `docker-compose up -d`.

# Example output file structure

Simba is heavily relying on the local file system as a transportation
layer to pass data from one step/function to another. Below is an
example of an output file structure and its files. In this example,

- one classifier: `mating`
- one source video: `tmp.mp4`
- in the video:
  - 300 frames
  - two mice, labeled as `animal_1` and `animal_2`


```plain
.
├── models
│   └── generated_models
│       ├── mating_meta.csv
│       ├── mating.sav
│       └── model_evaluations
│           ├── mating_classificationReport.png
│           ├── matingfancy_decision_tree_example
│           ├── matingfancy_decision_tree_example.svg
│           ├── mating_precision_recall.csv
│           ├── mating_tree.dot
│           └── mating_tree.pdf
└── project_folder
    ├── configs
    │   └── mating_meta_0.csv
    ├── csv
    │   ├── features_extracted
    │   │   └── tmp.csv
    │   ├── input_csv
    │   │   ├── original_filename
    │   │   │   └── tmp.csv
    │   │   └── tmp.csv
    │   ├── machine_results
    │   │   └── tmp.csv
    │   ├── outlier_corrected_movement
    │   │   └── tmp.csv
    │   ├── outlier_corrected_movement_location
    │   │   └── tmp.csv
    │   ├── targets_inserted
    │   │   └── tmp.csv
    │   └── validation
    │       └── tmp.csv
    ├── frames
    │   ├── input
    │   │   └── tmp
    │   │       ├── 0.png
    │   │       ├── 100.png
    │   │       ├── 101.png
    │   │       ├── 102.png
    │   │       ├── 103.png
    │   │       ├── 104.png
    │   │       ├── 105.png
    │   │       ├── 106.png
    │   │       ├── 107.png
    │   │       ├── 108.png
    │   │       ├── 109.png
    │   │       ├── 10.png
    │   │       ├── 110.png
    │   │       ├── 111.png
    │   │       ├── 112.png
    │   │       ├── 113.png
    │   │       ├── 114.png
    │   │       ├── 115.png
    │   │       ├── 116.png
    │   │       ├── 117.png
    │   │       ├── 118.png
    │   │       ├── 119.png
    │   │       ├── 11.png
    │   │       ├── 120.png
    │   │       ├── 121.png
    │   │       ├── 122.png
    │   │       ├── 123.png
    │   │       ├── 124.png
    │   │       ├── 125.png
    │   │       ├── 126.png
    │   │       ├── 127.png
    │   │       ├── 128.png
    │   │       ├── 129.png
    │   │       ├── 12.png
    │   │       ├── 130.png
    │   │       ├── 131.png
    │   │       ├── 132.png
    │   │       ├── 133.png
    │   │       ├── 134.png
    │   │       ├── 135.png
    │   │       ├── 136.png
    │   │       ├── 137.png
    │   │       ├── 138.png
    │   │       ├── 139.png
    │   │       ├── 13.png
    │   │       ├── 140.png
    │   │       ├── 141.png
    │   │       ├── 142.png
    │   │       ├── 143.png
    │   │       ├── 144.png
    │   │       ├── 145.png
    │   │       ├── 146.png
    │   │       ├── 147.png
    │   │       ├── 148.png
    │   │       ├── 149.png
    │   │       ├── 14.png
    │   │       ├── 150.png
    │   │       ├── 151.png
    │   │       ├── 152.png
    │   │       ├── 153.png
    │   │       ├── 154.png
    │   │       ├── 155.png
    │   │       ├── 156.png
    │   │       ├── 157.png
    │   │       ├── 158.png
    │   │       ├── 159.png
    │   │       ├── 15.png
    │   │       ├── 160.png
    │   │       ├── 161.png
    │   │       ├── 162.png
    │   │       ├── 163.png
    │   │       ├── 164.png
    │   │       ├── 165.png
    │   │       ├── 166.png
    │   │       ├── 167.png
    │   │       ├── 168.png
    │   │       ├── 169.png
    │   │       ├── 16.png
    │   │       ├── 170.png
    │   │       ├── 171.png
    │   │       ├── 172.png
    │   │       ├── 173.png
    │   │       ├── 174.png
    │   │       ├── 175.png
    │   │       ├── 176.png
    │   │       ├── 177.png
    │   │       ├── 178.png
    │   │       ├── 179.png
    │   │       ├── 17.png
    │   │       ├── 180.png
    │   │       ├── 181.png
    │   │       ├── 182.png
    │   │       ├── 183.png
    │   │       ├── 184.png
    │   │       ├── 185.png
    │   │       ├── 186.png
    │   │       ├── 187.png
    │   │       ├── 188.png
    │   │       ├── 189.png
    │   │       ├── 18.png
    │   │       ├── 190.png
    │   │       ├── 191.png
    │   │       ├── 192.png
    │   │       ├── 193.png
    │   │       ├── 194.png
    │   │       ├── 195.png
    │   │       ├── 196.png
    │   │       ├── 197.png
    │   │       ├── 198.png
    │   │       ├── 199.png
    │   │       ├── 19.png
    │   │       ├── 1.png
    │   │       ├── 200.png
    │   │       ├── 201.png
    │   │       ├── 202.png
    │   │       ├── 203.png
    │   │       ├── 204.png
    │   │       ├── 205.png
    │   │       ├── 206.png
    │   │       ├── 207.png
    │   │       ├── 208.png
    │   │       ├── 209.png
    │   │       ├── 20.png
    │   │       ├── 210.png
    │   │       ├── 211.png
    │   │       ├── 212.png
    │   │       ├── 213.png
    │   │       ├── 214.png
    │   │       ├── 215.png
    │   │       ├── 216.png
    │   │       ├── 217.png
    │   │       ├── 218.png
    │   │       ├── 219.png
    │   │       ├── 21.png
    │   │       ├── 220.png
    │   │       ├── 221.png
    │   │       ├── 222.png
    │   │       ├── 223.png
    │   │       ├── 224.png
    │   │       ├── 225.png
    │   │       ├── 226.png
    │   │       ├── 227.png
    │   │       ├── 228.png
    │   │       ├── 229.png
    │   │       ├── 22.png
    │   │       ├── 230.png
    │   │       ├── 231.png
    │   │       ├── 232.png
    │   │       ├── 233.png
    │   │       ├── 234.png
    │   │       ├── 235.png
    │   │       ├── 236.png
    │   │       ├── 237.png
    │   │       ├── 238.png
    │   │       ├── 239.png
    │   │       ├── 23.png
    │   │       ├── 240.png
    │   │       ├── 241.png
    │   │       ├── 242.png
    │   │       ├── 243.png
    │   │       ├── 244.png
    │   │       ├── 245.png
    │   │       ├── 246.png
    │   │       ├── 247.png
    │   │       ├── 248.png
    │   │       ├── 249.png
    │   │       ├── 24.png
    │   │       ├── 250.png
    │   │       ├── 251.png
    │   │       ├── 252.png
    │   │       ├── 253.png
    │   │       ├── 254.png
    │   │       ├── 255.png
    │   │       ├── 256.png
    │   │       ├── 257.png
    │   │       ├── 258.png
    │   │       ├── 259.png
    │   │       ├── 25.png
    │   │       ├── 260.png
    │   │       ├── 261.png
    │   │       ├── 262.png
    │   │       ├── 263.png
    │   │       ├── 264.png
    │   │       ├── 265.png
    │   │       ├── 266.png
    │   │       ├── 267.png
    │   │       ├── 268.png
    │   │       ├── 269.png
    │   │       ├── 26.png
    │   │       ├── 270.png
    │   │       ├── 271.png
    │   │       ├── 272.png
    │   │       ├── 273.png
    │   │       ├── 274.png
    │   │       ├── 275.png
    │   │       ├── 276.png
    │   │       ├── 277.png
    │   │       ├── 278.png
    │   │       ├── 279.png
    │   │       ├── 27.png
    │   │       ├── 280.png
    │   │       ├── 281.png
    │   │       ├── 282.png
    │   │       ├── 283.png
    │   │       ├── 284.png
    │   │       ├── 285.png
    │   │       ├── 286.png
    │   │       ├── 287.png
    │   │       ├── 288.png
    │   │       ├── 289.png
    │   │       ├── 28.png
    │   │       ├── 290.png
    │   │       ├── 291.png
    │   │       ├── 292.png
    │   │       ├── 293.png
    │   │       ├── 294.png
    │   │       ├── 295.png
    │   │       ├── 296.png
    │   │       ├── 297.png
    │   │       ├── 298.png
    │   │       ├── 299.png
    │   │       ├── 29.png
    │   │       ├── 2.png
    │   │       ├── 30.png
    │   │       ├── 31.png
    │   │       ├── 32.png
    │   │       ├── 33.png
    │   │       ├── 34.png
    │   │       ├── 35.png
    │   │       ├── 36.png
    │   │       ├── 37.png
    │   │       ├── 38.png
    │   │       ├── 39.png
    │   │       ├── 3.png
    │   │       ├── 40.png
    │   │       ├── 41.png
    │   │       ├── 42.png
    │   │       ├── 43.png
    │   │       ├── 44.png
    │   │       ├── 45.png
    │   │       ├── 46.png
    │   │       ├── 47.png
    │   │       ├── 48.png
    │   │       ├── 49.png
    │   │       ├── 4.png
    │   │       ├── 50.png
    │   │       ├── 51.png
    │   │       ├── 52.png
    │   │       ├── 53.png
    │   │       ├── 54.png
    │   │       ├── 55.png
    │   │       ├── 56.png
    │   │       ├── 57.png
    │   │       ├── 58.png
    │   │       ├── 59.png
    │   │       ├── 5.png
    │   │       ├── 60.png
    │   │       ├── 61.png
    │   │       ├── 62.png
    │   │       ├── 63.png
    │   │       ├── 64.png
    │   │       ├── 65.png
    │   │       ├── 66.png
    │   │       ├── 67.png
    │   │       ├── 68.png
    │   │       ├── 69.png
    │   │       ├── 6.png
    │   │       ├── 70.png
    │   │       ├── 71.png
    │   │       ├── 72.png
    │   │       ├── 73.png
    │   │       ├── 74.png
    │   │       ├── 75.png
    │   │       ├── 76.png
    │   │       ├── 77.png
    │   │       ├── 78.png
    │   │       ├── 79.png
    │   │       ├── 7.png
    │   │       ├── 80.png
    │   │       ├── 81.png
    │   │       ├── 82.png
    │   │       ├── 83.png
    │   │       ├── 84.png
    │   │       ├── 85.png
    │   │       ├── 86.png
    │   │       ├── 87.png
    │   │       ├── 88.png
    │   │       ├── 89.png
    │   │       ├── 8.png
    │   │       ├── 90.png
    │   │       ├── 91.png
    │   │       ├── 92.png
    │   │       ├── 93.png
    │   │       ├── 94.png
    │   │       ├── 95.png
    │   │       ├── 96.png
    │   │       ├── 97.png
    │   │       ├── 98.png
    │   │       ├── 99.png
    │   │       ├── 9.png
    │   │       └── test.csv
    │   └── output
    │       ├── gantt_plots
    │       │   └── tmp
    │       ├── heatmap_behavior
    │       │   ├── tmp.mp4
    │       │   └── tmp.png
    │       ├── line_plot
    │       │   └── tmp
    │       ├── live_data_table
    │       │   └── tmp
    │       ├── merged
    │       │   └── tmp.mp4
    │       ├── path_plots
    │       │   └── tmp
    │       ├── sklearn_results
    │       │   ├── tmp
    │       │   └── tmp.mp4
    │       └── validation
    │           ├── tmp_mating.avi
    │           └── tmp_mating_gantt.avi
    ├── logs
    │   ├── lastframe_log.ini
    │   ├── measures
    │   │   └── pose_configs
    │   │       └── bp_names
    │   ├── Movement_log_20210402205555.csv
    │   ├── Movement_log_20210404215129.csv
    │   ├── Outliers_location_20210402023339.csv
    │   ├── Outliers_location_20210402023709.csv
    │   ├── Outliers_location_20210402024006.csv
    │   ├── Outliers_movement_20210402023337.csv
    │   ├── Outliers_movement_20210402023707.csv
    │   ├── Outliers_movement_20210402024004.csv
    │   ├── severity_20210402203952.csv
    │   ├── sklearn_20210402205521.csv
    │   ├── Time_bins_ML_results_20210402205616.csv
    │   ├── Time_bins_movement_results_20210402205623.csv
    │   └── video_info.csv
    ├── project_config.ini
    └── videos
        └── tmp.mp4

36 directories, 340 files
```

# Global config

Settings driving the Simba computations are mostly saved in a global `/project_folder/project_config.ini` file. Its values can be categorized into three types:

1. Paths: These are tightly correlated to the file structure shown in
   the previous section. They are mostly represented as a `file
   selection` widget, whereas its value is saved in this file. These
   path values are used throughout code so function can acquire file
   as its data input. In essence, Simba is using file system as a data
   transportation layer, which makes tracking the data flow difficult.

   Because of this design choice, nearly all folder paths are
   pre-determined. Path for a particular filename will vary, for
   example, `mating.csv` is linked to a classifier called
   `mating`. However, its location, eg. `/generated_models/` is still
   fixed.

2. Computation settings: These values, eg. bin size, bout threshold,
   are set by user for his/her use case. In most cases they don't have
   a default value.

3. Housekeeping: Values used by Simba for its own housekeeping only,
   eg. project name.

An example of the config is shown below. This example has been through most of the workflow steps, thus you seen most values are filled out. Upon a new project, you should expect most values are blank.

- `/app/output/feng` is the root output folder designated to this run,
  thus having no significant meaning in the interest of computation
  except for its data inventory/grouping purpose.


```ini
[General settings]
project_path = /app/output/feng/project_folder
project_name = feng
csv_path = /app/output/feng/project_folder/csv
use_master_config = yes
config_folder = /app/output/feng/project_folder/configs
workflow_file_type = csv
animal_no = 2
os_system = Linux

[SML settings]
model_dir = /app/output/feng/models
model_path_1 = /app/output/feng/models/generated_models/mating.sav
no_targets = 1
target_name_1 = mating

[threshold_settings]
threshold_1 = 0.105

[Minimum_bout_lengths]
min_bout_1 = 20

[Frame settings]
frames_dir_in = /app/output/feng/project_folder/frames/input
frames_dir_out = /app/output/feng/project_folder/frames/output
mm_per_pixel =
distance_mm = 245

[Line plot settings]
bodyparts =

[Path plot settings]
deque_points = 1
behaviour_points =
plot_severity = no
severity_brackets = 10
file_format = .bmp
no_animal_pathplot = 2
animal_1_bp = Center_1
animal_2_bp = Center_2
severity_target = mating

[Frame folder]
frame_folder = /app/output/feng/project_folder/frames
copy_frames = yes

[Distance plot]
poi_1 = Center_1
poi_2 = Center_2

[Heatmap settings]
bin_size_pixels = 200
scale_max_seconds = auto
scale_increments_seconds =
palette = gnuplot2
target_behaviour =
body_part = Center_1
target = mating

[Heatmap location]

[ROI settings]
animal_1_bp =
animal_2_bp =
directionality_data =
visualize_feature_data =

[process movements]
animal_1_bp = Ear_left_1
animal_2_bp = Ear_left_2
no_of_animals = 2

[Create movie settings]
file_format =
bitrate =

[create ensemble settings]
pose_estimation_body_parts = 16
pose_config_label_path = /app/output/feng/project_folder/logs/measures/pose_configs/bp_names/project_bp_names.csv
model_to_run = RF
load_model =
data_folder = /app/output/feng/project_folder/csv/targets_inserted
classifier = mating
train_test_size = 0.2
under_sample_setting = None
under_sample_ratio = NaN
over_sample_setting = None
over_sample_ratio = NaN
rf_n_estimators = 2000
rf_min_sample_leaf = 1
rf_max_features = sqrt
rf_n_jobs = -1
rf_criterion = gini
rf_meta_data = yes
generate_example_decision_tree = yes
generate_example_decision_tree_fancy = yes
generate_features_importance_log = no
generate_features_importance_bar_graph = no
compute_permutation_importance = no
generate_learning_curve = no
generate_precision_recall_curve = yes
n_feature_importance_bars = NaN
gbc_n_estimators =
gbc_max_features =
gbc_max_depth =
gbc_learning_rate =
gbc_min_sample_split =
xgb_n_estimators =
xgb_max_depth =
xgb_learning_rate =
meta_files_folder = /app/output/feng/project_folder/configs
learningcurve_shuffle_k_splits = NaN
learningcurve_shuffle_data_splits = NaN
generate_classification_report = yes
generate_shap_scores = no
shap_target_present_no =
shap_target_absent_no =

[validation/run model]
generate_validation_video =
sample_feature_file =
save_individual_frames =
classifier_path =
classifier_name =
frames_dir_out_validation =
save_frames =
save_gantt =
discrimination_threshold =

[Multi animal IDs]
id_list =

[Outlier settings]
movement_criterion = 0.7
location_criterion = 1.5
movement_bodypart1_animal_1 = Ear_left_1
movement_bodypart2_animal_1 = Ear_right_1
location_bodypart1_animal_1 = Ear_left_1
location_bodypart2_animal_1 = Ear_right_1
movement_bodypart1_animal_2 = Ear_left_2
movement_bodypart2_animal_2 = Ear_right_2
location_bodypart1_animal_2 = Ear_left_2
location_bodypart2_animal_2 = Ear_right_2
mean_or_median = mean
```

# GUI step to function mapping

The goal is to map GUI steps to simba code so we could understand the
scope of its workflow, and possibility to skip GUI by running a script
directly to achieve the same workflow result.

## Create new project

TBD

## Load project

Once a project has been created, all file structure and configuration
baseline would have been created. Now we are to load project ini and
follow the [tutorial][1] step by step.

The `Load Project` button is defined as:

```python
# button
launchloadprojectButton = Button(
    lpMenu,
    text="Load Project",
    command=lambda: self.launch(lpMenu, inputcommand),
)
```

The `inputcommand` is actually a `loadprojectini` class, and the
`self.launch` command is defined as below. Thus it is taking the
select `.ini` file as the input to initialize a `loadprojectini` obj.

```python
def launch(self, master, command):

    # close parent GUI window
    master.destroy()

    if self.projectconfigini.file_path.endswith(".ini"):
        print(self.projectconfigini.file_path)
        command(self.projectconfigini.file_path)

    else:
        print("Please select the project_config.ini file")

```

This whole thing can then be shortened as below to achieve the same effect:

```python
MY_INI = "/app/output/feng/project_folder/project_config.ini"
master.destroy()
command(MY_INI)
return
```


## Further imports

If no further data points or inputs are needed, you can safely skip
this step.

![screen shot](/doc/images/further%20imports.png)

## Video parameters

The idea is to compute **pixels per millimeter**. This is achieved by

1. Knowing the resolution of the video of both width & height in unit
   of pixel.
2. Select two points &rarr; thus knowing the number of pixels between
   the two.
3. Tell the computer this length is mapped to a physical distance in
   real life w/ unit in `mm`.

So the equation would then be:

```plain
pixel per mm = physical length (mm) / # of pixels
```

This number, of course, will then be used later to map image measures
in pixels back into unit `mm`.

### Autopopulate table

![autopopulate table button](https://github.com/sgoldenlab/simba/raw/master/images/setvidparameter.PNG)

In class `loadprojectini`, line 5195, defined the button and its input:

```python
self.distanceinmm = Entry_Box(label_setscale, "Known distance (mm)", "18")
button_setdistanceinmm = Button(
    label_setscale,
    text="Autopopulate table",
    command=lambda: self.set_distancemm(self.distanceinmm.entry_get),
)
```

The button click will then trigger the call to `self.set_distancemm`, which is defined as:

```python
def set_distancemm(self, distancemm):

    configini = self.projectconfigini
    config = ConfigParser()
    config.read(configini)

    config.set("Frame settings", "distance_mm", distancemm)
    with open(configini, "w") as configfile:
        config.write(configfile)

```

Therefore, all its doing is to write the value in the input box, which is a physical distance in `mm`, to the config ini file:

```ini
[Frame settings]
frames_dir_in = /app/output/feng/project_folder/frames/input
frames_dir_out = /app/output/feng/project_folder/frames/output
mm_per_pixel =
distance_mm = 245 <== here
```

### Save Data

![video parameter save data](https://github.com/sgoldenlab/simba/raw/master/images/videoinfo_table2.PNG)

After user has drawn two dots on an image, thus telling the computer
the distance in pixel, code will compute the `pixels/mm` shown in the
input box. Now clicking on the `Save data` would trigger a call to the
`generate_video_info_csv`.

```python
generate_csv_button = Button(
    self.xscrollbar,
    text="Save Data",
    command=self.generate_video_info_csv,
    font="bold",
    fg="red",
)
```

And what does this `generate_video_info_csv` do? It does some
computations and dump the results in `logs/video_info.csv`.

```python
def generate_video_info_csv(self):
    # get latest data from table
    self.data_lists = []
    # get all data from tables
    for i in range(len(self.table_col)):
        self.data_lists.append([])
        for j in range(len(self.filesFound)):
            self.data_lists[i].append(self.table_col[i].entry_get(j))
    # get the ppm from table
    self.ppm = ["pixels/mm"]
    for i in range(len(self.filesFound) - 1):
        self.ppm.append((self.button.getppm(i)))
    self.data_lists.append(self.ppm)

    # remove .mp4 from first column
    self.data_lists[1] = [i.replace(i[-4:], "") for i in self.data_lists[1]]
    self.data_lists[1][0] = "Video"

    data = self.data_lists
    df = pd.DataFrame(data=data)
    df = df.transpose()
    df = df.rename(columns=df.iloc[0])
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(["index"], axis=1)
    df = df.drop(["level_0"], axis=1)

    logFolder = os.path.join(os.path.dirname(self.configFile), "logs")
    csv_filename = "video_info.csv"
    output = os.path.join(logFolder, csv_filename)

    df.to_csv(str(output), index=False)
    print(os.path.dirname(output), "generated.")

```

## Outlier correction

![outlier correction ui](https://github.com/sgoldenlab/simba/raw/master/images/outliercorrection.PNG)

### Outlier settings `Confirm`

![outlier correction Confirm button](https://github.com/sgoldenlab/simba/raw/master/images/outliercorrection2.PNG)

The `Confirm` button is defined as:

```python
button_setvalues = Button(
    scroll,
    text="Confirm",
    command=self.set_outliersettings,
    font=("Arial", 16, "bold"),
    fg="red",
)
```

And the definition of `self.set_outliersettings` is below. It does no
calculation but to grab the form values and write them into the config
file.

```python

def set_outliersettings(self):
    # export settings to config ini file
    configini = self.configini
    config = ConfigParser()
    config.read(configini)
    animalno = config.getint("General settings", "animal_no")
    animalNameList = []
    try:
        multiAnimalIDList = config.get("Multi animal IDs", "id_list")
        multiAnimalIDList = multiAnimalIDList.split(",")

    except NoSectionError:
        multiAnimalIDList = [""]

    if multiAnimalIDList[0] == "":
        for animal in range(animalno):
            animalNameList.append("Animal_" + str(animal + 1))
    else:
        animalNameList = multiAnimalIDList

    try:
        for animal in range(len(animalNameList)):
            locBp1 = self.var1List[animal].get()
            locBp2 = self.var2List[animal].get()
            movBp1 = self.var1ListMov[animal].get()
            movBp2 = self.var2ListMov[animal].get()
            config.set(
                "Outlier settings",
                "movement_bodyPart1_" + str(animalNameList[animal]),
                str(movBp1),
            )
            config.set(
                "Outlier settings",
                "movement_bodyPart2_" + str(animalNameList[animal]),
                str(movBp2),
            )
            config.set(
                "Outlier settings",
                "location_bodyPart1_" + str(animalNameList[animal]),
                str(locBp1),
            )
            config.set(
                "Outlier settings",
                "location_bodyPart2_" + str(animalNameList[animal]),
                str(locBp2),
            )
        movementcriterion = self.movement_criterion.entry_get
        locationcriterion = self.location_criterion.entry_get
        mean_or_median = self.medianvar.get()
        config.set("Outlier settings", "movement_criterion", str(movementcriterion))
        config.set("Outlier settings", "location_criterion", str(locationcriterion))
        config.set("Outlier settings", "mean_or_median", str(mean_or_median))

        with open(configini, "w") as configfile:
            config.write(configfile)
        print("Outlier correction settings updated in project_config.ini")
    except:
        print("Please make sure all fields are filled in correctly.")

```

An example `ini` section:

```ini
[Outlier settings]
movement_criterion = 0.7
location_criterion = 1.5
movement_bodypart1_animal_1 = Ear_left_1
movement_bodypart2_animal_1 = Ear_right_1
location_bodypart1_animal_1 = Ear_left_1
location_bodypart2_animal_1 = Ear_right_1
movement_bodypart1_animal_2 = Ear_left_2
movement_bodypart2_animal_2 = Ear_right_2
location_bodypart1_animal_2 = Ear_left_2
location_bodypart2_animal_2 = Ear_right_2
mean_or_median = mean
```

### Run outlier correction

The button is defined as:

```python
button_outliercorrection = Button(
    label_outliercorrection,
    text="Run outlier correction",
    command=self.correct_outlier,
)
```

And the `self.correct_outlier` is defined as:

```python
def correct_outlier(self):
    configini = self.projectconfigini
    config = ConfigParser()
    config.read(configini)
    pose_estimation_body_parts = config.get(
        "create ensemble settings", "pose_estimation_body_parts"
    )
    print(
        "Pose-estimation body part setting for outlier correction: "
        + str(pose_estimation_body_parts)
    )
    if (pose_estimation_body_parts == "16") or (
        pose_estimation_body_parts == "987"
    ):
        dev_move_16(configini)
        dev_loc_16(configini)
    if pose_estimation_body_parts == "14":
        dev_move_14(configini)
        dev_loc_14(configini)
    if (
        (pose_estimation_body_parts == "user_defined")
        or (pose_estimation_body_parts == "4")
        or (pose_estimation_body_parts == "7")
        or (pose_estimation_body_parts == "8")
        or (pose_estimation_body_parts == "9")
    ):
        dev_move_user_defined(configini)
        dev_loc_user_defined(configini)
    print("Outlier correction complete.")

```

Here we can see function `dev_mov_` and `dev_loc_` being used, and
these are defined in `correct_devs_mov_16bp.py` and
`correct_locs_move_16bp.py` respectively.

## Extract Features

![extract features](/doc/images/extract%20features.png)

The button is defined as:

```python
button_extractfeatures = Button(
    label_extractfeatures,
    text="Extract Features",
    command=lambda: threading.Thread(target=self.extractfeatures).start(),
)
```

So this will start a new thread running the `self.extractfeatures` func:

```python
def extractfeatures(self):
    configini = self.projectconfigini
    config = ConfigParser()
    config.read(configini)
    pose_estimation_body_parts = config.get(
        "create ensemble settings", "pose_estimation_body_parts"
    )
    print(
        "Pose-estimation body part setting for feature extraction: "
        + str(pose_estimation_body_parts)
    )
    userFeatureScriptStatus = self.usVar.get()
    print(userFeatureScriptStatus)

    if userFeatureScriptStatus == 1:
        pose_estimation_body_parts == "user_defined_script"
        import sys

        script = self.scriptfile.file_path
        print(script)
        dir = os.path.dirname(script)
        fscript = os.path.basename(script).split(".")[0]
        sys.path.insert(0, dir)
        import importlib

        mymodule = importlib.import_module(fscript)
        mymodule.extract_features_userdef(self.projectconfigini)

    if userFeatureScriptStatus == 0:
        if pose_estimation_body_parts == "16":
            extract_features_wotarget_16(self.projectconfigini)
        if pose_estimation_body_parts == "14":
            extract_features_wotarget_14(self.projectconfigini)
        if pose_estimation_body_parts == "987":
            extract_features_wotarget_14_from_16(self.projectconfigini)
        if pose_estimation_body_parts == "9":
            extract_features_wotarget_9(self.projectconfigini)
        if pose_estimation_body_parts == "8":
            extract_features_wotarget_8(self.projectconfigini)
        if pose_estimation_body_parts == "7":
            extract_features_wotarget_7(self.projectconfigini)
        if pose_estimation_body_parts == "4":
            extract_features_wotarget_4(self.projectconfigini)
        if pose_estimation_body_parts == "user_defined":
            extract_features_wotarget_user_defined(self.projectconfigini)

```

And these `extract_features_` functions are imported from `simba.feature_scripts` as:

```python
from simba.features_scripts.extract_features_4bp import \
    extract_features_wotarget_4
from simba.features_scripts.extract_features_7bp import \
    extract_features_wotarget_7
from simba.features_scripts.extract_features_8bp import \
    extract_features_wotarget_8
from simba.features_scripts.extract_features_9bp import \
    extract_features_wotarget_9
from simba.features_scripts.extract_features_14bp import \
    extract_features_wotarget_14
from simba.features_scripts.extract_features_14bp_from_16bp import \
    extract_features_wotarget_14_from_16
from simba.features_scripts.extract_features_16bp import \
    extract_features_wotarget_16
from simba.features_scripts.extract_features_user_defined import \
    extract_features_wotarget_user_defined

```

These functions do Panda computation on Euclidian, probability,
percentile, and so on, thus appears to be the heavy lifting utility.

## Label Behavior

![label behavior](https://github.com/sgoldenlab/simba/raw/master/images/label_behaviornew.PNG)

This is a manual process where operator is expected to identify
patterns s/he is looking for frame by frame, thus allowing AI to use
these as baseline for its training. An example 10s video contains 300
frames. So it's imaginable how lengthy and demanding this step can be
if human is to label _all_ the frames.

Check details of the [behavioral annotator GUI][2].

![labeling behavior GUI](https://github.com/sgoldenlab/simba/raw/master/images/labelling_mainscreen.PNG)

### select folder w/ frames

This is to set the path to video frames. Frames would have been
generated from video earlier, thus expecting to see a list of `.png`
files in the selected folder.

The button `Select folder with frames` is defined as:

```python
button_labelaggression = Button(
    label_labelaggression,
    text="Select folder with frames (create new video annotation)",
    command=lambda: choose_folder(self.projectconfigini),
)
```

where the `choose_folder()` is defined in
`labelling_aggression.py`. This function counts the number of `.png`
files in the selected folder, which is always in pattern of `<output
path>/<proj name>/project_folder/frames/input/<video name>/`.

```python
# Opens a dialogue box where user chooses folder with video frames for
# labelling, then loads the config file, creates an empty dataframe
# and loads the interface, Prints working directory, current video
# name, and number of frames in folder.
def choose_folder(project_name):
    global current_video, projectini
    projectini = project_name
    img_dir = filedialog.askdirectory()
    os.chdir(img_dir)
    dirpath = os.path.basename(os.getcwd())
    current_video = dirpath
    print("Loading video " + current_video + "...")

    global frames_in
    frames_in = []
    for i in os.listdir(os.curdir):
        if i.__contains__(".png"):
            frames_in.append(i)
        reset()

    frames_in = sorted(frames_in, key=lambda x: int(x.split(".")[0]))
    # print(frames_in)
    number_of_frames = len(frames_in)
    print("Number of Frames: " + str(number_of_frames))

    configure(project_name)

    MainInterface()

```

Function `configure()` is defined in `labelling_aggression.py` as:

```python
# Retrieves behavior names from the config file
def configure(file_name):
    config = ConfigParser()
    config.read(file_name)
    number_of_targets = config.get("SML settings", "No_targets")

    for i in range(1, int(number_of_targets) + 1):
        target = config.get("SML settings", "target_name_" + str(i))
        columns.append(target)
        behaviors.append(0)
        df[columns[i - 1]] = 0

```
where `columns`, `behaviors` and `df` are defined as global variables in the file:

```python
behaviors = []
columns = []
df = pd.DataFrame(index=new_index, columns=columns)
```

And example values in the `ini` are:

```ini
[SML settings]
model_dir = /app/output/feng/models
model_path_1 = /app/output/feng/models/generated_models/mating.sav
no_targets = 1
target_name_1 = mating
```

Note that if `no_targets=2`, the following lines would be like:

```
target_name_1 = mating
target_name_2 = sth
```
And this goes the same for the `model_path_` above also.

The `MainInterface` is defined also in `labelling_agression.py`. It
defines the GUI where user can label frame by frame of the applicable
classifier, and navigate through these frames using arrows keys on the
keyboard.

### Saven and advance to the next frame

This is used to label  one frame. Button is defined as:

```python
save = Button(
    self.window,
    text="Save and advance to the next frame",
    command=lambda: self.save_checkboxes(self.window),
)
```

And the `self.save_checkboxes` is defined as:

```python
def save_checkboxes(self, master):
    if self.rangeOn.get():
        s = int(self.firstFrame.get())
        e = int(self.lastFrame.get())
        save_values(s, e)
        if e < len(frames_in) - 1:
            load_frame(e + 1, master, self.fbox)
        else:
            load_frame(e, master, self.fbox)
    else:
        s = current_frame_number
        e = s
        save_values(s, e)
        load_frame(e + 1, master, self.fbox)
```

where `s` and `e` will be frame index number.

The key in this is the `save_values()`. What it does is to set values in the `df` based on frame index, and checked behavior/classifier.


```python
# Saves the values of each behavior in the DataFrame and prints out the
# updated data frame
def save_values(start, end):
    global columns
    contprintLoop = True
    print("\n")
    if start == end:
        for i in range(len(behaviors)):
            df.at[current_frame_number, columns[i]] = int(behaviors[i])
            if behaviors[i] != 0:
                print(
                    "Annotated behavior: " +
                    columns[i] + ". Frame: " + str(start) + "."
                )

    if start != end:
        for i in range(start, end + 1):
            for b in range(len(behaviors)):
                df.at[i, columns[b]] = int(behaviors[b])
                if behaviors[b] != 0 and (contprintLoop == True):
                    print(
                        "Annotated behavior: "
                        + columns[b]
                        + ". Start frame: "
                        + str(start)
                        + ". End frame: "
                        + str(end)
                    )
            contprintLoop = False

```

### Generate / Save csv

This button is defined as:

```python
self.generate = Button(
    self.window,
    text="Generate / Save csv",
    command=lambda: save_video(self.window),
)
```

And the `save_video()` essentially takes the dataframe in memory and dump it to file system. The complexity of this action is caused by the `resume` capability of this labelling exercise, in which one user can label certain number of frames, save, then load and resume. Therefore, each time this button is clicked, a `/logs/lastframe_log.ini` is also created, in which the last saved frame index is remembered. The final result is then saved in `/csv/targets_inserted/<video name>.csv`.


```python
# Appends data to corresponding features_extracted csv and exports as new csv
def save_video(master):
    input_file = (
        str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2])
        + r"\csv\features_extracted\\"
        + current_video
        + ".csv"
    )
    output_file = (
        str(os.path.split(os.path.dirname(os.path.dirname(os.getcwd())))[-2])
        + r"\csv\targets_inserted\\"
        + current_video
        + ".csv"
    )
    data = pd.read_csv(input_file)
    new_data = pd.concat([data, df], axis=1)
    new_data = new_data.fillna(0)
    new_data.rename(columns={"Unnamed: 0": "scorer"}, inplace=True)
    try:
        new_data.to_csv(output_file, index=FALSE)
        print(output_file)
        print('Annotation file for "' + str(current_video) + '"' + " created.")
        # saved last frame number on
        frameLog = os.path.join(
            os.path.dirname(projectini), "logs", "lastframe_log.ini"
        )
        if not os.path.exists(frameLog):
            f = open(frameLog, "w+")
            f.write("[Last saved frames]\n")
            f.close()

        config = ConfigParser()
        config.read(frameLog)
        config.set("Last saved frames", str(current_video), str(current_frame_number))
        # write
        with open(frameLog, "w") as configfile:
            config.write(configfile)

    except PermissionError:
        print(
            "You don not have permission to save the annotation file - check that the file is not open in a different application. If you are working of a server make sure the file is not open on a different computer."
        )
```

And an example of the `lastframe_log.ini`:

```ini
[Last saved frames]
tmp = 51
```

## Train Machine Models


## Load Metadata

### Load (meta CSV)

The `Load` button is defined as:

```python
load_data = Button(
    load_data_frame, text="Load", command=self.load_RFvalues, fg="blue"
)
```

And the `load_RFvalues` is defined as:

```python
def load_RFvalues(self):

    metadata = pd.read_csv(str(self.load_choosedata.file_path), index_col=False)
    # metadata = metadata.drop(['Feature_list'], axis=1)
    for m in metadata.columns:
        self.meta_dict[m] = metadata[m][0]
    print("Meta data file loaded")

    for key in self.meta_dict:
        cur_list = key.lower().split(sep="_")
        # print(cur_list)
        for i in self.settings:
            string = i.lblName.cget("text").lower()
            if all(map(lambda w: w in string, cur_list)):
                i.entry_set(self.meta_dict[key])
        for k in self.check_settings:
            string = k.cget("text").lower()
            if all(map(lambda w: w in string, cur_list)):
                if self.meta_dict[key] == "yes":
                    k.select()
                elif self.meta_dict[key] == "no":
                    k.deselect()

```

What this does is to load `/models/generated_models/<classifier>_meta.csv` and populate the form. An example of this csv is shown below, and together showing the UI form corresponding to this CSV:

![load metadata form](/doc/images/load%20metadata.png)

```csv
Classifier_name,RF_criterion,RF_max_features,RF_min_sample_leaf,RF_n_estimators,compute_feature_permutation_importance,generate_classification_report,generate_example_decision_tree,generate_features_importance_bar_graph,generate_features_importance_log,generate_precision_recall_curves,generate_rf_model_meta_data_file,generate_sklearn_learning_curves,learning_curve_data_splits,learning_curve_k_splits,n_feature_importance_bars,over_sample_ratio,over_sample_setting,train_test_size,under_sample_ratio,under_sample_setting
mating,gini,sqrt,1,2000,no,yes,yes,no,no,yes,yes,no,NaN,NaN,NaN,NaN,None,0.2,NaN,NaN
```

This leads to the next section, `Save settings into global environment`.

### Save settings into global environment

The button is defined as:

```python
button_settings_to_ini = Button(
    trainmms,
    text="Save settings into global environment",
    font=("Helvetica", 18, "bold"),
    fg="blue",
    command=self.set_values,
)
```

And the `set_values()` is defined below. Pretty straightforward. It takes input form values and dump them into the `project_config.ini`.

```python
def set_values(self):
    self.get_checkbox()
    #### settings
    model = self.var.get()
    n_estimators = self.label_nestimators.entry_get
    max_features = self.label_maxfeatures.entry_get
    criterion = self.label_criterion.entry_get
    test_size = self.label_testsize.entry_get
    min_sample_leaf = self.label_minsampleleaf.entry_get
    under_s_c_v = self.label_under_s_correctionvalue.entry_get
    under_s_settings = self.label_under_s_settings.entry_get
    over_s_ratio = self.label_over_s_ratio.entry_get
    over_s_settings = self.label_over_s_settings.entry_get
    classifier_settings = self.varmodel.get()

    # export settings to config ini file
    configini = self.configini
    config = ConfigParser()
    config.read(configini)

    config.set("create ensemble settings", "model_to_run", str(model))
    config.set("create ensemble settings", "RF_n_estimators", str(n_estimators))
    config.set("create ensemble settings", "RF_max_features", str(max_features))
    config.set("create ensemble settings", "RF_criterion", str(criterion))
    config.set("create ensemble settings", "train_test_size", str(test_size))
    config.set(
        "create ensemble settings", "RF_min_sample_leaf", str(min_sample_leaf)
    )
    config.set("create ensemble settings", "under_sample_ratio", str(under_s_c_v))
    config.set(
        "create ensemble settings", "under_sample_setting", str(under_s_settings)
    )
    config.set("create ensemble settings", "over_sample_ratio", str(over_s_ratio))
    config.set(
        "create ensemble settings", "over_sample_setting", str(over_s_settings)
    )
    config.set("create ensemble settings", "classifier", str(classifier_settings))
    config.set("create ensemble settings", "RF_meta_data", str(self.rfmetadata))
    config.set(
        "create ensemble settings",
        "generate_example_decision_tree",
        str(self.generate_example_d_tree),
    )
    config.set(
        "create ensemble settings",
        "generate_classification_report",
        str(self.generate_classification_report),
    )
    config.set(
        "create ensemble settings",
        "generate_features_importance_log",
        str(self.generate_features_imp_log),
    )
    config.set(
        "create ensemble settings",
        "generate_features_importance_bar_graph",
        str(self.generate_features_bar_graph),
    )
    config.set(
        "create ensemble settings",
        "N_feature_importance_bars",
        str(self.n_importance),
    )
    config.set(
        "create ensemble settings",
        "compute_permutation_importance",
        str(self.compute_permutation_imp),
    )
    config.set(
        "create ensemble settings",
        "generate_learning_curve",
        str(self.generate_learning_c),
    )
    config.set(
        "create ensemble settings",
        "generate_precision_recall_curve",
        str(self.generate_precision_recall_c),
    )
    config.set(
        "create ensemble settings",
        "LearningCurve_shuffle_k_splits",
        str(self.learningcurveksplit),
    )
    config.set(
        "create ensemble settings",
        "LearningCurve_shuffle_data_splits",
        str(self.learningcurvedatasplit),
    )
    config.set(
        "create ensemble settings",
        "generate_example_decision_tree_fancy",
        str(self.generate_example_decision_tree_fancy),
    )
    config.set(
        "create ensemble settings", "generate_shap_scores", str(self.getshapscores)
    )
    config.set(
        "create ensemble settings", "shap_target_present_no", str(self.shappresent)
    )
    config.set(
        "create ensemble settings", "shap_target_absent_no", str(self.shapabsent)
    )

    with open(configini, "w") as configfile:
        config.write(configfile)

    print("Settings exported to project_config.ini")

```

And an example in the config `ini` (Note that the values below include
more than what the meta is setting. They are shown here for
completeness of this section of the ini.)

```ini
[create ensemble settings]
pose_estimation_body_parts = 16
pose_config_label_path = /app/output/feng/project_folder/logs/measures/pose_configs/bp_names/project_bp_names.csv
model_to_run = RF
load_model =
data_folder = /app/output/feng/project_folder/csv/targets_inserted
classifier = mating
train_test_size = 0.2
under_sample_setting = None
under_sample_ratio =
over_sample_setting = None
over_sample_ratio =
rf_n_estimators = 2000
rf_min_sample_leaf = 1
rf_max_features = sqrt
rf_n_jobs = -1
rf_criterion = gini
rf_meta_data = yes
generate_example_decision_tree = yes
generate_example_decision_tree_fancy = yes
generate_features_importance_log = no
generate_features_importance_bar_graph = no
compute_permutation_importance = no
generate_learning_curve = no
generate_precision_recall_curve = yes
n_feature_importance_bars =
gbc_n_estimators =
gbc_max_features =
gbc_max_depth =
gbc_learning_rate =
gbc_min_sample_split =
xgb_n_estimators =
xgb_max_depth =
xgb_learning_rate =
meta_files_folder = /app/output/feng/project_folder/configs
learningcurve_shuffle_k_splits =
learningcurve_shuffle_data_splits =
generate_classification_report = yes
generate_shap_scores = no
shap_target_present_no =
shap_target_absent_no =
```

- RF N Estimators: 2000
- RF Max features: sqrt
- RF Criterion: gini
- Train Test Size: 0.2
- RF Min sample leaf: 1
- Under sample setting: None
- Under sample ratio:
- Over sample setting: None
- Over sample ratio:

- Model evaluations settings: 1,2,3,4, generate precision recall curves, <save settings to global>

Then, click "train single model form global environ".

### Save settings for specific model

Similar to saving settings to global env. This is just a way dump the same information but to a different meta file, one per classifier in the `/configs/<classifier>_meta_<index>.csv`.

The index is simply incremented every time you save based on the
existing `meta` number found in the `/configs`. For example, if the folder already has `_meta_2.csv`, the new setting will then be saved to `_meta_3.csv`, and so on.

The purpose of this capability is to allow user have multiple copies
of forms, each having some different combination and values. Later on,
one could load one particular form without having to go through the
setup again, whereas the copy in the global env is only one of many.

The button is defined as:

```python
button_save_meta = Button(
    trainmms,
    text="Save settings for specific model",
    font=("Helvetica", 18, "bold"),
    fg="green",
    command=self.save_new,
)
```

And the `save_new()` is defined as:

```python
def save_new(self):
    self.get_checkbox()
    meta_number = 0
    for f in os.listdir(os.path.join(os.path.dirname(self.configini), "configs")):
        if f.__contains__("_meta") and f.__contains__(str(self.varmodel.get())):
            meta_number += 1

    # for s in self.settings:
    #     meta_df[s.lblName.cget('text')] = [s.entry_get]
    new_meta_dict = {
        "RF_n_estimators": self.label_nestimators.entry_get,
        "RF_max_features": self.label_maxfeatures.entry_get,
        "RF_criterion": self.label_criterion.entry_get,
        "train_test_size": self.label_testsize.entry_get,
        "RF_min_sample_leaf": self.label_minsampleleaf.entry_get,
        "under_sample_ratio": self.label_under_s_correctionvalue.entry_get,
        "under_sample_setting": self.label_under_s_settings.entry_get,
        "over_sample_ratio": self.label_over_s_ratio.entry_get,
        "over_sample_setting": self.label_over_s_settings.entry_get,
        "generate_rf_model_meta_data_file": self.rfmetadata,
        "generate_example_decision_tree": self.generate_example_d_tree,
        "generate_classification_report": self.generate_classification_report,
        "generate_features_importance_log": self.generate_features_imp_log,
        "generate_features_importance_bar_graph": self.generate_features_bar_graph,
        "n_feature_importance_bars": self.n_importance,
        "compute_feature_permutation_importance": self.compute_permutation_imp,
        "generate_sklearn_learning_curves": self.generate_learning_c,
        "generate_precision_recall_curves": self.generate_precision_recall_c,
        "learning_curve_k_splits": self.learningcurveksplit,
        "learning_curve_data_splits": self.learningcurvedatasplit,
        "generate_shap_scores": self.getshapscores,
        "shap_target_present_no": self.shappresent,
        "shap_target_absetn_no": self.shapabsent,
    }
    meta_df = pd.DataFrame(new_meta_dict, index=[0])
    meta_df.insert(0, "Classifier_name", str(self.varmodel.get()))

    if currentPlatform == "Windows":
        output_path = (
            os.path.dirname(self.configini)
            + "\\configs\\"
            + str(self.varmodel.get())
            + "_meta_"
            + str(meta_number)
            + ".csv"
        )

    if currentPlatform == "Linux":
        output_path = (
            os.path.dirname(self.configini)
            + "/configs/"
            + str(self.varmodel.get())
            + "_meta_"
            + str(meta_number)
            + ".csv"
        )

    print(os.path.basename(str(output_path)), "saved")

    meta_df.to_csv(output_path, index=FALSE)

```

### Clear cache

This is simply to remove previously generated meta CSV files from `/configs`.

```python
def clearcache(self):
    configs_dir = os.path.join(os.path.dirname(self.configini), "configs")
    filelist = [f for f in os.listdir(configs_dir) if f.endswith(".csv")]
    for f in filelist:
        os.remove(os.path.join(configs_dir, f))
        print(f, "deleted")
```

This, however, also illustrates the current design that it uses the
file system and its predefined file structure to hold values, thus
behaving like a cache. In our attempt, however, we are to abolish this
method and use DB as backend for data persistence.

### Train single model from global environment

The button is defined as:

```python
button_trainmachinemodel = Button(
    label_trainmachinemodel,
    text="Train single model from global environment",
    fg="blue",
    command=lambda: threading.Thread(
        target=trainmodel2(self.projectconfigini)
    ).start(),
)
```

Thus this kicks off a thread calling `trainmodel2()` for the job.  The
only input is the global config file.

### Train multiple model

If you recall that we could save multiple sets of meta settings, this is easily perceivable &mdash; it enumerates th e meta configs, and apply them to the data.

The button is defined as:

```python
button_train_multimodel = Button(
    label_trainmachinemodel,
    text="Train multiple models, one for each saved settings",
    fg="green",
    command=lambda: threading.Thread(target=self.trainmultimodel).start(),
)

```

## Validate model on Single video

- select feature file: select `/csv/features_extracted/<video name>.csv`
- select model: `/models/generate_models/<classfier>.csv`

### Run Model

The button is defined as:

```python
button_runvalidmodel = Button(
    label_model_validation,
    text="Run Model",
    command=lambda: validate_model_one_vid_1stStep(
        self.projectconfigini, self.csvfile.file_path, self.modelfile.file_path
    ),
)
```

And the function is defined in `runmodel_1st.py` as:

```python
def validate_model_one_vid_1stStep(inifile,csvfile,savfile):
    configFile = str(inifile)
    config = ConfigParser()
    config.read(configFile)
    sample_feature_file = str(csvfile)
    sample_feature_file_Name = os.path.basename(sample_feature_file)
    sample_feature_file_Name = sample_feature_file_Name.split('.', 1)[0]
    classifier_path = savfile
    classifier_name = os.path.basename(classifier_path).replace('.sav','')
    inputFile = pd.read_csv(sample_feature_file)
    inputFile = inputFile.loc[:, ~inputFile.columns.str.contains('^Unnamed')]
    outputDf = inputFile
    inputFileOrganised = drop_bp_cords(inputFile, inifile)
    print(inputFileOrganised)
    print('Running model...')
    clf = pickle.load(open(classifier_path, 'rb'))
    ProbabilityColName = 'Probability_' + classifier_name
    predictions = clf.predict_proba(inputFileOrganised)
    outputDf[ProbabilityColName] = predictions[:, 1]

    # CREATE LIST OF GAPS BASED ON SHORTEST BOUT

    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    fps = vidinfDf.loc[vidinfDf['Video'] == str(sample_feature_file_Name.replace('.csv', ''))]
    try:
        fps = int(fps['fps'])
    except TypeError:
        print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')


    outFname = sample_feature_file_Name + '.csv'
    csv_dir_out_validation = config.get('General settings', 'csv_path')
    csv_dir_out_validation = os.path.join(csv_dir_out_validation,'validation')
    if not os.path.exists(csv_dir_out_validation):
        os.makedirs(csv_dir_out_validation)
    outFname = os.path.join(csv_dir_out_validation, outFname)
    outputDf.to_csv(outFname)
    print('Predictions generated.')

```
### Generate plot

This will display a line plot show frame vs. <classifier> probability. It constructs the UI so that user can double click the line plot and the video window will jump to the particular frame, thus allowing operator to view the probability value and frame side-by-side.

The button is defined as:

```python
button_generateplot = Button(
    label_model_validation, text="Generate plot", command=self.updateThreshold
)
```

And the `updateThreshold()` is defined as:

```python
def updateThreshold(self):
    updateThreshold_graph(
        self.projectconfigini, self.csvfile.file_path, self.modelfile.file_path
    )
```

And the `updateThreshold_graph()` is defined in `prob_graph.py`. I'm
omitting the details of this function as much of it is to construct
the graph window and the video frame window, and how they are
responding to user clicks.

### Validate

The button is defined as:

```python
button_validate_model = Button(
    label_model_validation, text="Validate", command=self.validatemodelsinglevid
)
```

The `validatemodelsinglevid()` is defined below, nothing but a thin wrapper to the `validate_model_on_vid()`. Note that the two input values, `dis_threshold` and `min_behaviorbout` are required.

```python
def validatemodelsinglevid(self):
    validate_model_one_vid(
        self.projectconfigini,
        self.csvfile.file_path,
        self.modelfile.file_path,
        self.dis_threshold.entry_get,
        self.min_behaviorbout.entry_get,
        self.ganttvar.get(),
    )
```

`validate_model_one_vid()` is defined in `validate_model_on_single_video.py`. Results will be saved in `/frames/output/validation/`.


### Mode Settings

#### Model Settings

Model settings brings up a new GUI in which each classifier can be set a `(threshold, minimum bout)` value pairs.

The `Set model(s)` button is defined as:

```python
button_set = Button(
    runmms,
    text="Set model(s)",
    command=lambda: self.set_modelpath_to_ini(inifile),
    font=("Helvetica", 18, "bold"),
    fg="red",
)
```

The `set_modelpath_to_ini()` is defined as:

```python
def set_modelpath_to_ini(self, inifile):
    config = ConfigParser()
    configini = str(inifile)
    config.read(configini)

    for i in range(len(self.targetname)):
        config.set(
            "SML settings",
            "model_path_" + (str(i + 1)),
            str(self.row2[i].file_path),
        )
        config.set(
            "threshold_settings",
            "threshold_" + (str(i + 1)),
            str(self.row3[i].get()),
        )
        config.set(
            "Minimum_bout_lengths",
            "min_bout_" + (str(i + 1)),
            str(self.row4[i].get()),
        )

    with open(configini, "w") as configfile:
        config.write(configfile)

    print("Model paths saved in project_config.ini")

```

And example of the ini settings are:

```ini
[SML settings]
model_dir = /app/output/feng/models
model_path_1 = /app/output/feng/models/generated_models/mating.sav
no_targets = 1
target_name_1 = mating

[threshold_settings]
threshold_1 = 0.105

[Minimum_bout_lengths]
min_bout_1 = 20
```

Essentially, for each classifier, a model setting includes three things:

1. `model_path_<classifier index>`
2. `threshold_<classifier index>`
3. `min_bout_<classifier index>`

### Run RF Model

The button is defined as:

```python
button_runmachinemodel = Button(
    label_runmachinemodel,
    text="Run RF Model",
    command=lambda: threading.Thread(target=self.runrfmodel).start(),
)
```

So this is another thread and the main body is the `runrfmodel()`, which is nothing but a thin wrapper:

```python
def runrfmodel(self):
    rfmodel(self.projectconfigini)
```

The `rfmodel()` is defined in `run_RF_model.py`. Results are written in `/csv/machine_results/`.

### Apply Kleinburg Smoother

Button is defined as:

```python
run_kleinberg_button = Button(
    kleintoplvl,
    text="Apply Kleinberg Smoother",
    command=lambda: self.runkleinberg(targetlist, varlist),
)
```

And the `self.runkleinberg()` is defined as:

```python
def runkleinberg(self, targetlist, varlist):
    classifier_list = []
    for i in range(len(varlist)):
        if varlist[i].get() == 1:
            classifier_list.append(targetlist[i])

    print(classifier_list, "selected")
    run_kleinberg(
        self.projectconfigini,
        classifier_list,
        self.k_sigma.entry_get,
        self.k_gamma.entry_get,
        self.k_hierarchy.entry_get,
    )
```

However, though the import indicates that `from
simba.Kleinberg_burst_analysis import run_kleinberg`, I have failed to
identify the `Kleinberg_burst_analysis` module in the source
code. Thus this could either be a bug or a 3rd party installation.

### Analyze machine predictions

Skipping the popup GUI asking user to select a list of checkboxes. The actual action button `Analyze` is defined as:

```python
button1 = Button(
    dlmlabel,
    text="Analyze",
    command=lambda: self.findDatalogList(titlebox, var),
)
```

And the `findDatalogList()` is defined as:

```python
def findDatalogList(self, titleBox, Var):
    finallist = []
    for index, i in enumerate(Var):
        if i.get() == 0:
            finallist.append(titleBox[index])

    # run analyze
    analyze_process_data_log(self.projectconfigini, finallist)
```

Func `analyze_process_data_log()` is defined in `process_data_log.py`. This is where the calculation for `bout events` takes place, which later produce the overlay info:

1. total events duration (s)
2. mean bout duration (s)
3. median bout duration (s)
4. first occurance (s)
5. mean interval (s)
6. median interval (s)

### Analyze distance/velocity

Action button `Run` on this UI (rendered by `roi_settings()`) is
defined as:

```python
runButton = Button(
    self.secondMenu,
    text=text,
    command=lambda: self.run_analyze_roi(
        noofanimal.get(), animalVarList, appendornot
    ),
)
```

And the `run_analyze_roi()` is defined as:

```python
def run_analyze_roi(self, noofanimal, animalVarList, appendornot):
    print(animalVarList)
    configini = self.projectconfigini
    config = ConfigParser()
    config.read(configini)

    if appendornot == "processmovement":
        config.set("process movements", "no_of_animals", str(noofanimal))
        for animal in range(noofanimal):
            animalBp = str(animalVarList[animal].get())
            config.set(
                "process movements", "animal_" + str(animal + 1) + "_bp", animalBp
            )
        with open(configini, "w") as configfile:
            config.write(configfile)

    elif appendornot == "locationheatmap":
        animalBp = str(self.animalbody1var.get())
        config.set("Heatmap location", "body_part", animalBp)
        config.set("Heatmap location", "Palette", str(self.pal_var.get()))
        config.set(
            "Heatmap location", "Scale_max_seconds", str(self.scalemaxsec.entry_get)
        )
        config.set(
            "Heatmap location", "bin_size_pixels", str(self.binsizepixels.entry_get)
        )
        with open(configini, "w") as configfile:
            config.write(configfile)

    else:
        config.set("ROI settings", "no_of_animals", str(noofanimal))
        for animal in range(noofanimal):
            currStr = "animal_" + str(animal + 1) + "_bp"
            config.set("ROI settings", currStr, str(animalVarList[animal].get()))
            with open(configini, "w") as configfile:
                config.write(configfile)

    if appendornot == "append":
        ROItoFeatures(configini)
    elif appendornot == "not append":
        config.set(
            "ROI settings",
            "probability_threshold",
            str(self.p_threshold_a.entry_get),
        )
        roiAnalysis(configini, "outlier_corrected_movement_location")
    elif appendornot == "processmovement":
        ROI_process_movement(configini)
    elif appendornot == "locationheatmap":
        plotHeatMapLocation(
            configini,
            animalBp,
            int(self.binsizepixels.entry_get),
            str(self.scalemaxsec.entry_get),
            self.pal_var.get(),
            self.lastimgvar.get(),
        )
    elif appendornot == "direction":
        print("ROI settings saved.")
    else:
        roiAnalysis(configini, "features_extracted")
```

It's messy, isn't it!? Results are saved in `/logs`.


### Time bins: Machine predictions

The action button is defined as:

```python
tb_button = Button(
    tb_labelframe,
    text="Run",
    command=lambda: time_bins_classifier(
        self.projectconfigini, int(tb_entry.entry_get)
    ),
)
```

And `time_bins_classifier()` is defined in `timBins_classifier.py`. Results are written in `/logs/Time_bins_ML_results_<now in %Y%m%d%H%M%S>.csv`.

### Time bins: Distance/Velocity

Similar to the "time bins: machine prediction" above, button is defined as:

```python
tb_button = Button(
    tb_labelframe,
    text="Run",
    command=lambda: time_bins_movement(
        self.projectconfigini, int(tb_entry.entry_get)
    ),
)

```

The `time_bins_movement(0` is defined in `timeBins_movement.py`. Results are written in `/logs/Time_bins_movement_results_<time stamp>.csv`.

### Analyze target severity

Button is defined as:

```python
button_process_severity = Button(
    label_severity, text="Analyze target severity", command=self.analyzseverity
)
```

And the `self.analyzeseverity()` is a thin wrapper:

```python
def analyzseverity(self):
    analyze_process_severity(
        self.projectconfigini,
        self.severityscale.entry_get,
        self.severityTarget.getChoices(),
    )
```

Func `analyze_process_severity()` is defined in `process_severity.py`. Results are written in `/logs/severity_<now time stamp>.csv`.


[1]: https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md
[2]: https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md
