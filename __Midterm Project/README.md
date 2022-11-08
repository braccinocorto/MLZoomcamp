# Heart Stroke prediction

## Problem description
Heart stroke looks like unpredictable. Is there a chance to predict what are the most probable sources that lead to an heart stroke?
Will it be age? The place you live in? Previous heart diseases?
We try to analyze this dataset and see if we can have a prediction based on some feature analysis.
The model intends to predict if, given a set of features, what's the probability for that individual to get stroke.

This is a binary classification project.

The dataset used to train the model is taken from kaggle:
url: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv
[download][linkdata]

The target value is the column "stroke", which is binary.

## Features
- id: the row id
- gender: Male/Female
- age: integer, the age of the individual
- hypertension: the person has hypertension (1/0: Yes/No)
- heart_disease: the person has already had an heart disease during his lifetime (1/0: Yes/No)
- ever_married: Yes/No
- work_type: Cildren, Government Job, Never worked, Private, Self-Employee
- Residence_type: Urban/Rural
- avg_glucose_level : float, an index of the average glucose found in the blood
- bmi: float, body mass index.
- smoking_status: smokes / formerly smoked / never smoked / Unknown
- stroke: 1/0 if it has had an heart stroke or not. This is our target variable.


## EDA
In the notebook, you can find 
- Classification of the features
- Distribution of values of the features
- Correlation Matrix on the raw data, including categorical values

!(./imgs/corr_matrix.jpg "Correlaion Matrix")

We finally see that the most correlated variables are age, hyertension, average glucose level.
Quite surprisingly, smoke does not look so correlated. And so does work type.


## Model Training
- Split the dataset in 80/20/20 train/validation/test
- The training process aims to compare the AUC value for the regressors analyzed. So that we can compare them on the same field.
- Train a Logistic Regressor. I've tested the different regression algos with different C values. The best combination (maximising AUC) results in the ldfgs solver, with C parameter set to 1.
- Train a Randomforest. I've explored the combinations of n_estimators and max_depth. The best combination is reached with maxdepth = 10 and 170 estimators,
- Train XGBoost. To find the best combination, through capture output (with an iteratos number of 500, we test different  values of eta, max depth, min child weight). Through iterative testing, the best combination found is eta=0.2, max_depth=2, min_child_weight=10.
- Evaluation of the best performing model [via AUC_score confrontation]
- Selection of the model: XGBoost.

Through xgb.plot_tree and xgb.plot_importance, we find a confirmation of the initial correlation we found from the data.
The three main features are avg_glucoselevel, bmi and age.
Hypertension does not appear in the most important features, accoridng to the model.

!(/imgs/features_importance.jpg "Features Importance")

!(/imgs/features_importance_zoomin.jpg "Zoom in Features Importance")


## Exporting the notebook
- With the selected model, and hyperparameter tuning, I created the train.py file. For training data, it uses local csv file (the same as downloaded). In order to train an create the model (saved by bento) run
    python train.py
- Created a test.py so that we can load the model (via Bento), provide data (in JSON format) and receive a response (the json data to use to send the request is written into the test.py file. This tool is useful when testing locally and simply modify the data submitted with a txt editor and see the result from the CLI)
    python test.py
- Created a predict.py file, that loads the model previously trained, ready for the deployment. Included a pydantic model in order to prevent erroneous data.
- The output received from the model is a probability, and as such is treated (with a threshold of 0.5), even if some other level of risk are provided.

Based on the dataset, a value > than 0.5 provides a likely heartstroke outcome
0.20 is the threshold where the model predictions have a probability distribution that of the training data on the dataset. 
So, a value above the threshold may lead to a condition which is 'above the average' (based on the dataset)


## Environment
- To manage the environment dependencies and build the model file, I've used Bento.
- BentoML. I've created the Bento model file. This is realy easy with the bento package. Move in to the bentofile.yaml then 
    ```bash
    bentoml build bentofile.yaml
    ```

I've written the yaml file. The only file of the project included is the service.py file. The librares included are xgb, sklearn, pandas and pydantic. The service.py file loads the bento model previously generated.
- Run the model locally via BentoML serve command:
    ```bash
    bentoml serve predict.py:svc
    ```
And then run your test (manually) in the interface [JSON in, JSON out]:

!(./imgs/bento_serving_locally.jpg "BentoML swagger interface")

- Finally, the 'dockerization' of the project: launched docker for desktop (in order to start docker demon), and then 
    ```bash
    bentoml containerize heart_stroke_model:latest 
    ```

it took 1240 sec!

!(./imgs/bento_built.jpg "BentoML in all its awesomeness")

- The generated Docker includes a base debian distro (default)
- Finally re-run it locally in via docker container:
    ```bash
    docker run -it --rm -p 3000:3000 heart_stroke_model:latest serve --production
    ```

!(./imgs/docker_locally_running.jpg "Docker running the service stack locally")

- BentoML's container creator generates the dockerfile locally (in the folder bentos/[bento name]/[tag]/env/docker/Dockerfile )
The DockerFile you find in this folder is the one generated by the BentoML containerization.

## Deployment to the Cloud
- A first local test via Docker has been made and confirms the model running.
- I deployed the service to Google Cloud Run via bentoctl. 
- Set up the gcloud account, install gcloud CLI, install terraform and install bentoctl
- gcloud CLI is the package provided from Google in order to interact via CLI with the various services of GoogleCloud.
- terraform (CLI) is an interface to mae the deployment you're preparing platform agnostic. So that with a simple change, you can deploy your service to GCloud or AWS or Azure. Once set up the basics to interact with the cloud (authentication required), terraform handles all the backstage required to have the service deployed to a provider or another.
- Through a step by step interactive guide, bentoctl creates the files to use [deployment_config.yaml, main.tf, bentoctl.tfvars] [tf files are for terraforming]
- Via a simple bentoctl build, the tool creates the image and then uploads the image to cloudrun image registry.
- And then apply the deployment via Terraform. (where the .tf files generated are now put into use.). Also, if there are problems of memory limits imposed by the GCRun environment, it is possible to modify the TF files and re-deploy without need to rebuild the image and reupload it to the repository.
- The deployment pipeline is really smooth and ready for CI/CD
- The project has been uploaded and deployed at this address: https://quickstart-cloud-run-service-skes2kiqua-uc.a.run.app/
 [link][applink]


   [linkdata]: <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv>
   [applink]: <https://quickstart-cloud-run-service-skes2kiqua-uc.a.run.app/>