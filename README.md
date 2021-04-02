# mouse

## Dev & deployment

Pre-requisite: Linux, because simba is a GUI app, and would require X window feature to use the host's display.

Two dockers will be created: Simba is based on a `miniconda` base image, and `deeplabcut` on Python:3.7.

1. Install docker & docker-compose, eg. `pip install docker-compose`
2. For the first run: `docker-compose up --build -d`. This will take a while as it's pulling a lot of things from conda repo.
3. For consequetive runs, `docker-compose up -d simba` if you only want simba to run. For both deeplabcut & simba, `docker-compose up -d`.


TODO: below, write step-to-step wizard including:
- tab name
- input & its value
- button to click/action

Later on, we will follow these steps 1-by-1 to trace them down into
the code w/ a goal to bypass all the manual inputs and selections so
to use script to run the entire process w/o human intervention once a
config has been given.

## Load project config

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
