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

# GUI step to function mapping

The goal is to map GUI steps to simba code so we could understand the scope of its workflow, and possibility to skip GUI by running a script directly to achieve the same workflow result.

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

### Generate / Save annotations

TBD

## Notes

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

### Validate model on Single video

- select feature file: select `/csv/features.../tmp.csv`
- select model: `/feng/models/generate_models/mating.csv`

Click `run model`

[1]: https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md
[2]: https://github.com/sgoldenlab/simba/blob/master/docs/labelling_aggression_tutorial.md
