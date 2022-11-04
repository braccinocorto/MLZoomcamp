# Heart Stroke prediction

## Problem description
Heart stroke looks like unpredictable. We try to analyze this dataset and see if we can have a prediction based on some feature analysis.
The model intends to predict if, given a set of features, what's the probability for that individual to get stroke.

This is a binary classification project.

The dataset used to train the model is taken from kaggle:
url: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv
[download][linkdata]

The target value is the column "stroke", which is binary.

## EDA
- Classification of the features
- Distribution of values of the features
- Correlation Matrix on the raw data, including categorical values


## Model Training
- Split the dataset in 80/20/20 train/validation/test
- The training process aims to compare the AUC value for the regressors analyzed. So that we can compare them on the same field.
- Train a Logistic Regressor. I've tested the different regression algos with different C values. 
- Train a Randomforest. I've explored the combinations of n_estimators and max_depth.
- Train XGBoost. To find the best combination, through capture output (with an iteratos number of 500, we test differentv  values of eta, max depth, min child weight)
- Evaluation of the best performing model [via AUC_score confrontation]
- Selection of the model: XGBoost.

## Exporting the notebook
- With the selected model, and hyperparameter tuning, I created the train.py file. For training data, it uses local csv file (the same as downloaded).
- Created a test.py so that we can load the model (via Bento), provide data (in JSON format) and receive a response
- Created a predict.py file, based on BentoML, ready for the deployment. Included a pydantic model in order to prevent erroneous data.
- The output received from the model is a probability, and as such is treated (with a threshold of 0.5), even if some other level of risk are provided.
- Pydantic requirements to be added with the @validator decorators

## Environment
- BentoML. I've written the yaml file. The only file included is the service.py file. The librares included are xgb, sklearn, pandas and pydantic.
- I've then bentoml built the model. Which is  a compact 455 kb file.
- Finally, the 'dockerization' of the project: launched docker for desktop (in order to start docker demon), and then bentoml containerize heart_stroke_model:latest - it took 1240 sec!
- Docker includes a base debian distro (default)
- Finally re-run it in a docker container
- BentoML generates the dockerfile locally (in the folder bentos/[bento name]/[tag]/env/docker/Dockerfile )


## Deployment to the Cloud
- A first local test via Docker has been made and confirms the model running.
- I tried to deploy to Google Cloud Run via bentoctl. 
- Set up the gcloud account, install gcloud CLI, install terraform and install bentoctl
- Through a step by step guide, bentoctl creates the files to use [deployment_config.yaml, main.tf, bentoctl.tfvars] [tf files are for terraforming]
- Via a simple bentoctl build, the tool creates the image and then uploads the image to cloudrun image registry.
- And then apply the deployment via Terraform. (where the .tf files generated are now put into use.). Also, if there are problems of memory limits imposed by the GCRun environment, it is possible to modify the TF files and re-deploy without need to rebuild the image and reupload it to the repository.
- The deployment pipeline is really smooth and ready for CI/CD
- The project has been uploaded and deployed at this address: https://quickstart-cloud-run-service-skes2kiqua-uc.a.run.app/
 [link][applink]


   [linkdata]: <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download&select=healthcare-dataset-stroke-data.csv>
   [applink]: <https://quickstart-cloud-run-service-skes2kiqua-uc.a.run.app/>