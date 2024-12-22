import streamlit as st

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Test 1','Test 2','Test 3','Test 4','Test 5','Test 6'])

with tab1:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

    # Questions and their options
    questions = [
        {
            "question": "You've recently created a custom neural network that relies on essential dependencies unique to your organization's framework. Now, you want to train this model using a managed training service in Google Cloud. However, there's a challenge: the ML framework and its related dependencies aren't compatible with AI Platform Training. Additionally, both your model and data exceed the capacity of a single machine's memory. Your preferred ML framework is designed around a distribution structure involving schedulers, workers, and servers. What steps should you take in this situation?",
            "options": [
                "A. Use a built-in model available on AI Platform Training.",
                "B. Build your custom container to run jobs on AI Platform Training.",
                "C. Build your custom containers to run distributed training jobs on AI Platform Training.",
                "D. Reconfigure your code to a ML framework with dependencies that are supported by AI Platform Training."
            ],
            "answer": "C. Build your custom containers to run distributed training jobs on AI Platform Training."
        },
        {
            "question": "You're in charge of a data science team within a large international corporation. Your team primarily develops large-scale models using high-level TensorFlow APIs on AI Platform with GPUs. The typical iteration time for a new model version ranges from a few weeks to several months. Recently, there has been a request to assess and reduce your team's Google Cloud compute costs while ensuring that the model's performance remains unaffected. How can you achieve this cost reduction without compromising the model's quality?",
            "options": [
                "A. Use AI Platform to run distributed training jobs with checkpoints.",
                "B. Use AI Platform to run distributed training jobs without checkpoints.",
                "C. Migrate to training with Kubeflow on Google Kubernetes Engine, and use preemptible VMs with checkpoints.",
                "D. Migrate to training with Kubeflow on Google Kubernetes Engine, and use preemptible VMs without checkpoints."
            ],
            "answer": "C. Migrate to training with Kubeflow on Google Kubernetes Engine, and use preemptible VMs with checkpoints."
        },
        {
            "question": "You have deployed a model on Vertex AI for real-time inference. While processing an online prediction request, you encounter an 'Out of Memory' error. What should be your course of action?",
            "options": [
                "A. Use batch prediction mode instead of online mode.",
                "B. Send the request again with a smaller batch of instances.",
                "C. Use base64 to encode your data before using it for prediction.",
                "D. Apply for a quota increase for the number of prediction requests."
            ],
            "answer": "B. Send the request again with a smaller batch of instances."
        },
        {
            "question": "You are profiling the performance of your TensorFlow model training time and have identified a performance issue caused by inefficiencies in the input data pipeline. This issue is particularly evident when working with a single 5 terabyte CSV file dataset stored on Cloud Storage. What should be your initial action to improve the efficiency of your pipeline?",
            "options": [
                "A. Preprocess the input CSV file into a TFRecord file.",
                "B. Randomly select a 10 gigabyte subset of the data to train your model.",
                "C. Split into multiple CSV files and use a parallel interleave transformation.",
                "D. Set the reshuffle_each_iteration parameter to true in the tf.data.Dataset.shuffle method."
            ],
            "answer": "C. Split into multiple CSV files and use a parallel interleave transformation."
        },
        {
            "question": "You are logged into the Vertex AI Pipeline UI and noticed that an automated production TensorFlow training pipeline finished three hours earlier than a typical run. You do not have access to production data for security reasons, but you have verified that no alert was logged in any of the ML system’s monitoring systems and that the pipeline code has not been updated recently. You want to debug the pipeline as quickly as possible so you can determine whether to deploy the trained model. What should you do?",
            "options": [
                "A. Navigate to Vertex AI Pipelines, and open Vertex AI TensorBoard. Check whether the training regime and metrics converge.",
                "B. Access the Pipeline run analysis pane from Vertex AI Pipelines, and check whether the input configuration and pipeline steps have the expected values.",
                "C. Determine the trained model’s location from the pipeline’s metadata in Vertex ML Metadata, and compare the trained model’s size to the previous model.",
                "D. Request access to production systems. Get the training data’s location from the pipeline’s metadata in Vertex ML Metadata, and compare data volumes of the current run to the previous run."
            ],
            "answer": "A. Navigate to Vertex AI Pipelines, and open Vertex AI TensorBoard. Check whether the training regime and metrics converge."
        },
        {
            "question": "You downloaded a TensorFlow language model pre-trained on a proprietary dataset by another company, and you tuned the model with Vertex AI Training by replacing the last layer with a custom dense layer. The model achieves the expected offline accuracy; however, it exceeds the required online prediction latency by 20ms. You want to optimize the model to reduce latency while minimizing the offline performance drop before deploying the model to production. What should you do?",
            "options": [
                "A. Apply post-training quantization on the tuned model, and serve the quantized model.",
                "B. Use quantization-aware training to tune the pre-trained model on your dataset, and serve the quantized model.",
                "C. Use pruning to tune the pre-trained model on your dataset, and serve the pruned model after stripping it of training variables.",
                "D. Use clustering to tune the pre-trained model on your dataset, and serve the clustered model after stripping it of training variables."
            ],
            "answer": "A. Apply post-training quantization on the tuned model, and serve the quantized model."
        },
        {
            "question": "You recently used Vertex AI Prediction to deploy a custom-trained model in production. The automated re-training pipeline made available a new model version that passed all unit and infrastructure tests. You want to define a rollout strategy for the new model version that guarantees an optimal user experience with zero downtime. What should you do?",
            "options": [
                "A. Release the new model version in the same Vertex AI endpoint. Use traffic splitting in Vertex AI Prediction to route a small random subset of requests to the new version and, if the new version is successful, gradually route the remaining traffic to it.",
                "B. Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application.",
                "C. Deploy the current model version with an Istio resource in Google Kubernetes Engine, and route production traffic to it. Deploy the new model version, and use Istio to route a small random subset of traffic to it. If the new version is successful, gradually route the remaining traffic to it.",
                "D. Install Seldon Core and deploy an Istio resource in Google Kubernetes Engine. Deploy the current model version and the new model version using the multi-armed bandit algorithm in Seldon to dynamically route requests between the two versions before eventually routing all traffic over to the best-performing version."
            ],
            "answer": "B. Release the new model version in a new Vertex AI endpoint. Update the application to send all requests to both Vertex AI endpoints, and log the predictions from the new endpoint. If the new version is successful, route all traffic to the new application."
        },
        {
            "question": "You are developing an ML model using a dataset with categorical input variables. You have randomly split half of the data into training and test sets. After applying one-hot encoding on the categorical variables in the training set, you discover that one categorical variable is missing from the test set. What should you do?",
            "options": [
                "A. Use sparse representation in the test set.",
                "B. Randomly redistribute the data, with 70% for the training set and 30% for the test set.",
                "C. Apply one-hot encoding on the categorical variables in the test data.",
                "D. Collect more data representing all categories."
            ],
            "answer": "C. Apply one-hot encoding on the categorical variables in the test data."
        },
        {
            "question": "You have successfully trained a DNN regressor using TensorFlow to predict housing prices, utilizing a set of predictive features. The default precision for your model is tf.float64, and you've employed a standard TensorFlow estimator with the following configuration:\n\n```\nestimator = tf.estimator.DNNRegressor(\n   feature_columns=[YOUR_LIST_OF_FEATURES],\n   hidden_units=[1024, 512, 256],\n   dropout=None\n)\n```\n\nYour model's performance is satisfactory; however, as you prepare to deploy it into production, you notice that your current serving latency on CPUs is 10ms at the 90th percentile. Your production requirements dictate a model latency of 8ms at the 90th percentile, and you are open to a slight decrease in prediction performance to meet this latency requirement. To achieve this, what should be your initial approach to quickly reduce the serving latency?",
            "options": [
                "A. Switch from CPU to GPU serving.",
                "B. Apply quantization to your SavedModel by reducing the floating point precision to tf.float16.",
                "C. Increase the dropout rate to 0.8 and retrain your model.",
                "D. Increase the dropout rate to 0.8 in _PREDICT mode by adjusting the TensorFlow Serving parameters."
            ],
            "answer": "B. Apply quantization to your SavedModel by reducing the floating point precision to tf.float16."
        },
        {
            "question": "You work for a retailer that sells clothes to customers around the world. You have been tasked with ensuring that ML models are built in a secure manner. Specifically, you need to protect sensitive customer data that might be used in the models. You have identified four fields containing sensitive data that are being used by your data science team: AGE, IS_EXISTING_CUSTOMER, LATITUDE_LONGITUDE, and SHIRT_SIZE. What should you do with the data before it is made available to the data science team for training purposes?",
            "options": [
                "A. Tokenize all of the fields using hashed dummy values to replace the real values.",
                "B. Use principal component analysis (PCA) to reduce the four sensitive fields to one PCA vector.",
                "C. Coarsen the data by putting AGE into quantiles and rounding LATITUDE_LONGTTUDE into single precision. The other two fields are already as coarse as possible.",
                "D. Remove all sensitive data fields, and ask the data science team to build their models using non-sensitive data."
            ],
            "answer": "A. Tokenize all of the fields using hashed dummy values to replace the real values."
        },
        {
            "question": "You are a member of the AI team at an automotive company, and your current project involves building a visual defect detection model using TensorFlow and Keras. To enhance the performance of your model, you intend to integrate various image augmentation techniques, including translation, cropping, and contrast adjustments. These augmentation methods will be applied randomly to each training batch. Your objective is to optimize the data processing pipeline for both runtime efficiency and efficient utilization of computational resources. What steps should you take to achieve this goal?",
            "options": [
                "A. Embed the augmentation functions dynamically in the tf.Data pipeline.",
                "B. Embed the augmentation functions dynamically as part of Keras generators.",
                "C. Use Dataflow to create all possible augmentations, and store them as TFRecords.",
                "D. Use Dataflow to create the augmentations dynamically per training run, and stage them as TFRecords."
            ],
            "answer": "A. Embed the augmentation functions dynamically in the tf.Data pipeline."
        },
        {
            "question": "You are a member of a data science team at a bank, tasked with building an ML model for predicting loan default risk. Your dataset, consisting of hundreds of millions of cleaned records, is stored in a BigQuery table. Your objective is to create and evaluate multiple models using TensorFlow and Vertex AI while ensuring that the data ingestion process is efficient and scalable. To achieve this, what steps should you take to minimize bottlenecks during data ingestion?",
            "options": [
                "A. Use the BigQuery client library to load data into a dataframe, and use tf.data.Dataset.from_tensor_slices() to read it.",
                "B. Export data to CSV files in Cloud Storage, and use tf.data.TextLineDataset() to read them.",
                "C. Convert the data into TFRecords, and use tf.data.TFRecordDataset() to read them.",
                "D. Use TensorFlow I/O’s BigQuery Reader to directly read the data."
            ],
            "answer": "D. Use TensorFlow I/O’s BigQuery Reader to directly read the data."
        },
        {
            "question": "During the exploratory data analysis of a dataset, you've identified a crucial categorical feature with a 5% incidence of missing values. To mitigate potential bias stemming from these gaps in the data, what would be your recommended approach for handling these missing values?",
            "options": [
                "A. Remove the rows with missing values, and upsample your dataset by 5%.",
                "B. Replace the missing values with the feature’s mean.",
                "C. Replace the missing values with a placeholder category indicating a missing value.",
                "D. Move the rows with missing values to your validation dataset."
            ],
            "answer": "C. Replace the missing values with a placeholder category indicating a missing value."
        },
        {
            "question": "You have the task of designing a recommendation system for a new video streaming platform. Your goal is to suggest the next video for users to watch. After receiving approval from an AI Ethics team, you're ready to commence development. Although your company's video catalog contains valuable metadata (e.g., content type, release date, country), you currently lack historical user event data. How should you go about constructing the recommendation system for the initial product version?",
            "options": [
                "A. Launch the product without machine learning. Present videos to users alphabetically, and start collecting user event data so you can develop a recommender model in the future.",
                "B. Launch the product without machine learning. Use simple heuristics based on content metadata to recommend similar videos to users, and start collecting user event data so you can develop a recommender model in the future.",
                "C. Launch the product with machine learning. Use a publicly available dataset such as MovieLens to train a model using the Recommendations AI, and then apply this trained model to your data.",
                "D. Launch the product with machine learning. Generate embeddings for each video by training an autoencoder on the content metadata using TensorFlow. Cluster content based on the similarity of these embeddings, and then recommend videos from the same cluster."
            ],
            "answer": "B. Launch the product without machine learning. Use simple heuristics based on content metadata to recommend similar videos to users, and start collecting user event data so you can develop a recommender model in the future."
        },
        {
            "question": "You work as an ML engineer at an ecommerce company, and your current assignment involves constructing a model for forecasting the optimal monthly inventory orders for the logistics team. How should you proceed with this task?",
            "options": [
                "A. Use a clustering algorithm to group popular items together. Give the list to the logistics team so they can increase inventory of the popular items.",
                "B. Use a regression model to predict how much additional inventory should be purchased each month. Give the results to the logistics team at the beginning of the month so they can increase inventory by the amount predicted by the model.",
                "C. Use a time series forecasting model to predict each item's monthly sales. Give the results to the logistics team so they can base inventory on the amount predicted by the model.",
                "D. Use a classification model to classify inventory levels as UNDER_STOCKED, OVER_STOCKED, and CORRECTLY_STOCKED. Give the report to the logistics team each month so they can fine-tune inventory levels."
            ],
            "answer": "C. Use a time series forecasting model to predict each item's monthly sales. Give the results to the logistics team so they can base inventory on the amount predicted by the model."
        },
        {
            "question": "To analyze user activity data from your company's mobile applications using BigQuery for data analysis, transformation, and ML algorithm experimentation, you must establish real-time data ingestion into BigQuery. What steps should you take to achieve this?",
            "options": [
                "A. Configure Pub/Sub to stream the data into BigQuery.",
                "B. Run an Apache Spark streaming job on Dataproc to ingest the data into BigQuery.",
                "C. Run a Dataflow streaming job to ingest the data into BigQuery.",
                "D. Configure Pub/Sub and a Dataflow streaming job to ingest the data into BigQuery."
            ],
            "answer": "A. Configure Pub/Sub to stream the data into BigQuery."
        },
        {
            "question": "When you observe oscillations in the loss during batch training of a neural network, how should you modify your model to ensure convergence?",
            "options": [
                "A. Decrease the size of the training batch.",
                "B. Decrease the learning rate hyperparameter.",
                "C. Increase the learning rate hyperparameter.",
                "D. Increase the size of the training batch."
            ],
            "answer": "B. Decrease the learning rate hyperparameter."
        },
        {
            "question": "You are employed by a gaming company with millions of customers worldwide. Your games offer a real-time chat feature that enables players to communicate with each other in over 20 languages. These messages are translated in real time using the Cloud Translation API. Your task is to create an ML system that moderates the chat in real time while ensuring consistent performance across various languages, all without altering the serving infrastructure. You initially trained a model using an in-house word2vec model to embed the chat messages translated by the Cloud Translation API. However, this model exhibits notable variations in performance among different languages. How can you enhance the model's performance in this scenario?",
            "options": [
                "A. Add a regularization term such as the Min-Diff algorithm to the loss function.",
                "B. Train a classifier using the chat messages in their original language.",
                "C. Replace the in-house word2vec with GPT-3 or T5.",
                "D. Remove moderation for languages for which the false positive rate is too high."
            ],
            "answer": "A. Add a regularization term such as the Min-Diff algorithm to the loss function."
        },
        {
            "question": "You work for an organization that operates a cloud-based communication platform combining chat, voice, and video conferencing. The platform stores audio recordings with an 8 kHz sample rate, all lasting over a minute. You are tasked with implementing a feature that automatically transcribes voice call recordings into text for future applications like call summarization and sentiment analysis. How should you implement this voice call transcription feature according to Google-recommended best practices?",
            "options": [
                "A. Retain the original audio sampling rate and transcribe the audio using the Speech-to-Text API with synchronous recognition.",
                "B. Retain the original audio sampling rate and transcribe the audio using the Speech-to-Text API with asynchronous recognition.",
                "C. Upsample the audio recordings to 16 kHz and transcribe the audio using the Speech-to-Text API with synchronous recognition.",
                "D. Upsample the audio recordings to 16 kHz and transcribe the audio using the Speech-to-Text API with asynchronous recognition."
            ],
            "answer": "B. Retain the original audio sampling rate and transcribe the audio using the Speech-to-Text API with asynchronous recognition."
        },
        {
            "question": "You are currently in the process of training a machine learning model for object detection. Your dataset comprises approximately three million X-ray images, each with an approximate size of 2 GB. You have set up the training process using Vertex AI Training, utilizing a Compute Engine instance equipped with 32 cores, 128 GB of RAM, and an NVIDIA P100 GPU. However, you've observed that the model training process is taking an extended period. Your objective is to reduce the training time without compromising the model's performance. What steps should you take to achieve this?",
            "options": [
                "A. Increase the instance memory to 512 GB and increase the batch size.",
                "B. Replace the NVIDIA P100 GPU with a v3-32 TPU in the training job.",
                "C. Enable early stopping in your Vertex AI Training job.",
                "D. Use the tf.distribute.Strategy API and run a distributed training job."
            ],
            "answer": "B. Replace the NVIDIA P100 GPU with a v3-32 TPU in the training job."
        },
        {
            "question": "You are employed as an ML engineer at a social media company, and your current project involves creating a visual filter for users' profile photos. This entails training an ML model to identify bounding boxes around human faces. Your goal is to integrate this filter into your company's iOS-based mobile application with minimal code development while ensuring that the model is optimized for efficient inference on mobile devices. What steps should you take?",
            "options": [
                "A. Train a model using Vertex AI AutoML Vision and use the “export for Core ML” option.",
                "B. Train a model using Vertex AI AutoML Vision and use the “export for Coral” option.",
                "C. Train a model using Vertex AI AutoML Vision and use the “export for TensorFlow.js” option.",
                "D. Train a custom TensorFlow model and convert it to TensorFlow Lite (TFLite)."
            ],
            "answer": "A. Train a model using Vertex AI AutoML Vision and use the “export for Core ML” option."
        },
        {
            "question": "As an ML engineer at a bank, you've created a binary classification model using Vertex AI AutoML Tables to determine whether a customer will make timely loan payments, which is critical for loan approval decisions. Now, the bank's risk department has requested an explanation for why the model rejected a specific customer's loan application. What steps should you take in response to this request?",
            "options": [
                "A. Use local feature importance from the predictions.",
                "B. Use the correlation with target values in the data summary page.",
                "C. Use the feature importance percentages in the model evaluation page.",
                "D. Vary features independently to identify the threshold per feature that changes the classification."
            ],
            "answer": "A. Use local feature importance from the predictions."
        },
        {
            "question": "You are employed by a prominent social network service provider where users publish articles and engage in news discussions. With millions of comments posted daily and over 200 human moderators screening comments for appropriateness, your team is developing an ML model to assist these human moderators in content review. The model assigns scores to each comment and identifies suspicious ones for human review. Which metric(s) should be employed to monitor the model's performance?",
            "options": [
                "A. Number of messages flagged by the model per minute.",
                "B. Number of messages flagged by the model per minute confirmed as being inappropriate by humans.",
                "C. Precision and recall estimates based on a random sample of 0.1% of raw messages each minute sent to a human for review.",
                "D. Precision and recall estimates based on a sample of messages flagged by the model as potentially inappropriate each minute."
            ],
            "answer": "D. Precision and recall estimates based on a sample of messages flagged by the model as potentially inappropriate each minute."
        },
        {
            "question": "You've trained a deep neural network model on Google Cloud that shows low loss on training data but underperforms on validation data, indicating overfitting. What strategy should be adopted to enhance the model's resilience against overfitting during retraining?",
            "options": [
                "A. Apply a dropout parameter of 0.2, and decrease the learning rate by a factor of 10.",
                "B. Apply a L2 regularization parameter of 0.4, and decrease the learning rate by a factor of 10.",
                "C. Run a hyperparameter tuning job on AI Platform to optimize for the L2 regularization and dropout parameters.",
                "D. Run a hyperparameter tuning job on AI Platform to optimize for the learning rate, and increase the number of neurons by a factor of 2."
            ],
            "answer": "C. Run a hyperparameter tuning job on AI Platform to optimize for the L2 regularization and dropout parameters."
        },
        {
            "question": "You are training an LSTM-based model on Google Cloud AI Platform to summarize text. The job submission script is as follows: \n\n```bash\n    gcloud ai-platform jobs submit training $JOB_NAME \\\n    --package-path $TRAINER_PACKAGE_PATH \\\n    --module-name $MAIN_TRAINER_MODULE \\\n    --job-dir $JOB_DIR \\\n    --region $REGION \\\n    --scale-tier basic \\\n    -- \\\n    --epochs 20 \\\n    --batch_size=32 \\\n    --learning_rate=0.001 \\\n```\n\nYou want to ensure that training time is minimized without significantly compromising the accuracy of your model. What should you do?",
            "options": [
                "A. Modify the ‘epochs’ parameter.",
                "B. Modify the ‘scale-tier’ parameter.",
                "C. Modify the ‘batch_size’ parameter.",
                "D. Modify the ‘learning_rate’ parameter."
            ],
            "answer": "B. Modify the ‘scale-tier’ parameter."
        },
        {
            "question": "As an ML engineer at a worldwide shoe retailer, overseeing the company's website's machine learning models, you've been tasked with creating a recommendation model. This model should suggest new products to customers, taking into account their purchasing habits and similarities with other users. How should you proceed to build this model?",
            "options": [
                "A. Build a classification model",
                "B. Build a knowledge-based filtering model",
                "C. Build a collaborative-based filtering model",
                "D. Build a regression model using the features as predictors"
            ],
            "answer": "C. Build a collaborative-based filtering model"
        },
        {
            "question": "You are tasked with creating a unified analytics environment that spans across various on-premises data marts. The company faces data quality and security issues during data integration across servers, stemming from the use of diverse, disconnected tools and makeshift solutions. The goal is to adopt a fully managed, cloud-native data integration service that reduces overall workload costs and minimizes repetitive tasks. Additionally, some team members favor a codeless interface for constructing Extract, Transform, Load (ETL) processes. Which service would best meet these requirements?",
            "options": [
                "A. Dataflow",
                "B. Dataprep",
                "C. Apache Flink",
                "D. Cloud Data Fusion"
            ],
            "answer": "D. Cloud Data Fusion"
        },
        {
            "question": "Your team successfully trained and tested a DNN regression model, but six months post-deployment, its performance has declined due to changes in the input data distribution. What approach should you take to tackle these differences in the input data in the production environment?",
            "options": [
                "A. Create alerts to monitor for skew, and retrain the model.",
                "B. Perform feature selection on the model, and retrain the model with fewer features.",
                "C. Retrain the model, and select an L2 regularization parameter with a hyperparameter tuning service.",
                "D. Perform feature selection on the model, and retrain the model on a monthly basis with fewer features."
            ],
            "answer": "A. Create alerts to monitor for skew, and retrain the model."
        },
        {
            "question": "Your production demand forecasting pipeline preprocesses raw data using Dataflow before model training and prediction. This involves applying Z-score normalization to data in BigQuery and then writing it back. With new training data added weekly, your goal is to enhance efficiency by reducing both computation time and manual effort. What steps should you take to achieve this?",
            "options": [
                "A. Normalize the data using Google Kubernetes Engine.",
                "B. Translate the normalization algorithm into SQL for use with BigQuery.",
                "C. Use the normalizer_fn argument in TensorFlow's Feature Column API.",
                "D. Normalize the data with Apache Spark using the Dataproc connector for BigQuery."
            ],
            "answer": "B. Translate the normalization algorithm into SQL for use with BigQuery."
        },
        {
            "question": "You've trained a text classification model in TensorFlow on AI Platform and now need to perform batch predictions on text data stored in BigQuery, all while minimizing computational overhead. What's the recommended approach?",
            "options": [
                "A. Export the model to BigQuery ML.",
                "B. Deploy and version the model on AI Platform.",
                "C. Use Dataflow with the SavedModel to read the data from BigQuery.",
                "D. Submit a batch prediction job on AI Platform that points to the model location in Cloud Storage."
            ],
            "answer": "D. Submit a batch prediction job on AI Platform that points to the model location in Cloud Storage."
        },
        {
            "question": "You are an ML engineer at a global car manufacturer. Your task is to develop an ML model for predicting car sales in various cities worldwide. Which features or feature combinations should you use to capture city-specific relationships between car types and the number of sales?",
            "options": [
                "A. Three individual features: binned latitude, binned longitude, and one-hot encoded car type.",
                "B. One feature obtained as an element-wise product between latitude, longitude, and car type.",
                "C. One feature obtained as an element-wise product between binned latitude, binned longitude, and one-hot encoded car type.",
                "D. Two feature crosses as an element-wise product: the first between binned latitude and one-hot encoded car type, and the second between binned longitude and one-hot encoded car type."
            ],
            "answer": "C. One feature obtained as an element-wise product between binned latitude, binned longitude, and one-hot encoded car type."
        },
        {
            "question": "You are training a TensorFlow model on a structured dataset with 100 billion records stored in several CSV files. You need to improve the input/output execution performance. What should you do?",
            "options": [
                "A. Load the data into BigQuery, and read the data from BigQuery.",
                "B. Load the data into Cloud Bigtable, and read the data from Bigtable.",
                "C. Convert the CSV files into shards of TFRecords, and store the data in Cloud Storage.",
                "D. Convert the CSV files into shards of TFRecords, and store the data in the Hadoop Distributed File System."
            ],
            "answer": "C. Convert the CSV files into shards of TFRecords, and store the data in Cloud Storage."
        },
        {
            "question": "You started working on a classification problem with time series data and achieved an area under the receiver operating characteristic curve (AUC ROC) value of 99% for training data after just a few experiments. You haven't explored using any sophisticated algorithms or spent any time on hyperparameter tuning. What should your next step be to identify and fix the problem?",
            "options": [
                "A. Address the model overfitting by using a less complex algorithm.",
                "B. Address data leakage by applying nested cross-validation during model training.",
                "C. Address data leakage by removing features highly correlated with the target value.",
                "D. Address the model overfitting by tuning the hyperparameters to reduce the AUC ROC value."
            ],
            "answer": "C. Address data leakage by removing features highly correlated with the target value."
        },
        {
            "question": "You are aiming to train a deep learning model for semantic image segmentation with a focus on reducing training time. However, when using a Deep Learning VM Image, you encounter the following error: \n\n```\nThe resource 'projects/deeplearning-platform/zones/europe-west4-c/acceleratorTypes/nvidia-tesla-k80' was not found.\n```\n\n What steps should you take to address this issue?",
            "options": [
                "A. Ensure that you have GPU quota in the selected region.",
                "B. Ensure that the required GPU is available in the selected region.",
                "C. Ensure that you have preemptible GPU quota in the selected region.",
                "D. Ensure that the selected GPU has enough GPU memory for the workload."
            ],
            "answer": "B. Ensure that the required GPU is available in the selected region."
        },
        {
            "question": "You've recently become a part of a machine learning team that is on the verge of launching a new project. Taking on a lead role for this project, you've been tasked with assessing the production readiness of the machine learning components. The team has already conducted tests on features and data, completed model development, and ensured the readiness of the infrastructure. What further readiness check would you advise the team to consider?",
            "options": [
                "A. Ensure that training is reproducible.",
                "B. Ensure that all hyperparameters are tuned.",
                "C. Ensure that model performance is monitored.",
                "D. Ensure that feature expectations are captured in the schema."
            ],
            "answer": "C. Ensure that model performance is monitored."
        },
        {
            "question": "You need to train an object detection model to identify bounding boxes around Post-it Notes® in an image. Post-it Notes can have a variety of background colors and shapes. You have a dataset with 1000 images with a maximum size of 1.4MB and a CSV file containing annotations stored in Cloud Storage. You want to select a training method that reliably detects Post-it Notes of any relative size in the image and that minimizes the time to train a model. What should you do?",
            "options": [
                "A. Use the Cloud Vision API in Vertex AI with OBJECT_LOCALIZATION type, and filter the detected objects that match the Post-it Note category only.",
                "B. Upload your dataset into Vertex AI. Use Vertex AI AutoML Vision Object Detection with accuracy as the optimization metric, early stopping enabled, and no training budget specified.",
                "C. Write a Python training application that trains a custom vision model on the training set. Autopackage the application, and configure a custom training job in Vertex AI.",
                "D. Write a Python training application that performs transfer learning on a pre-trained neural network. Autopackage the application, and configure a custom training job in Vertex AI."
            ],
            "answer": "B. Upload your dataset into Vertex AI. Use Vertex AI AutoML Vision Object Detection with accuracy as the optimization metric, early stopping enabled, and no training budget specified."
        },
        {
            "question": "You recently developed a deep learning model. To test your new model, you trained it for a few epochs on a large dataset. You observe that the training and validation losses barely changed during the training run. You want to quickly debug your model. What should you do first?",
            "options": [
                "A. Verify that your model can obtain a low loss on a small subset of the dataset",
                "B. Add handcrafted features to inject your domain knowledge into the model",
                "C. Use the Vertex AI hyperparameter tuning service to identify a better learning rate",
                "D. Use hardware accelerators and train your model for more epochs"
            ],
            "answer": "A. Verify that your model can obtain a low loss on a small subset of the dataset"
        },
        {
            "question": "You have recently deployed a machine learning model, and it has come to your attention that after three months of deployment, the model is exhibiting underperformance on specific subgroups, raising concerns about potential bias in the results. This inequitable performance is suspected to be linked to class imbalances in the training data, and the option of collecting additional data is not available. In this situation, what steps should you take? (Select two options.)",
            "options": [
                "A. Remove training examples of high-performing subgroups, and retrain the model.",
                "B. Add an additional objective to penalize the model more for errors made on the minority class, and retrain the model",
                "C. Remove the features that have the highest correlations with the majority class.",
                "D. Upsample or reweight your existing training data, and retrain the model",
                "E. Redeploy the model, and provide a label explaining the model's behavior to users."
            ],
            "answer": [
                "B. Add an additional objective to penalize the model more for errors made on the minority class, and retrain the model",
                "D. Upsample or reweight your existing training data, and retrain the model"
            ]
        },
        {
            "question": "You are constructing a machine learning model for real-time anomaly detection in sensor data. To manage incoming requests, Pub/Sub will be utilized. The goal is to store the results for subsequent analytics and visualization. How should you configure the pipeline?",
            "options": [
                "A. 1 = Dataflow, 2 = AI Platform, 3 = BigQuery",
                "B. 1 = DataProc, 2 = Vertex AI AutoML, 3 = Cloud Bigtable",
                "C. 1 = BigQuery, 2 = Vertex AI AutoML, 3 = Cloud Functions",
                "D. 1 = BigQuery, 2 = AI Platform, 3 = Cloud Storage"
            ],
            "answer": "A. 1 = Dataflow, 2 = AI Platform, 3 = BigQuery"
        },
        {
            "question": "You were tasked with investigating failures in a production line component using sensor readings. Upon receiving the dataset, you found that less than 1% of the readings are positive examples representing failure incidents. Despite multiple attempts to train classification models, none of them have converged. How should you address the issue of class imbalance?",
            "options": [
                "A. Utilize the class distribution to generate 10% positive examples.",
                "B. Implement a convolutional neural network with max pooling and softmax activation.",
                "C. Downsample the data with upweighting to create a sample with 10% positive examples.",
                "D. Remove negative examples until the numbers of positive and negative examples are equal."
            ],
            "answer": "C. Downsample the data with upweighting to create a sample with 10% positive examples."
        },
        {
            "question": "You're creating machine learning models for CT scan image segmentation using AI Platform and regularly update the architectures to align with the latest research. To benchmark their performance, you need to retrain these models using the same dataset. Your goal is to reduce computational expenses and manual effort, while maintaining version control for your code. What steps should you take?",
            "options": [
                "A. Use Cloud Functions to identify changes to your code in Cloud Storage and trigger a retraining job.",
                "B. Use the gcloud command-line tool to submit training jobs on AI Platform when you update your code.",
                "C. Use Cloud Build linked with Cloud Source Repositories to trigger retraining when new code is pushed to the repository.",
                "D. Create an automated workflow in Cloud Composer that runs daily and looks for changes in code in Cloud Storage using a sensor."
            ],
            "answer": "C. Use Cloud Build linked with Cloud Source Repositories to trigger retraining when new code is pushed to the repository."
        },
        {
            "question": "Your team is tasked with developing a model to predict if images contain a driver’s license, passport, or credit card. The data engineering team has already constructed the pipeline and created a dataset, comprising 10,000 images with driver’s licenses, 1,000 images with passports, and 1,000 images with credit cards. Your objective now is to train a model using the following label map: ['drivers_license', 'passport', 'credit_card']. Which loss function is most suitable for this task?",
            "options": [
                "A. Categorical hinge",
                "B. Binary cross-entropy",
                "C. Categorical cross-entropy",
                "D. Sparse categorical cross-entropy"
            ],
            "answer": "D. Sparse categorical cross-entropy"
        },
        {
            "question": "You've utilized Vertex AI Workbench notebooks to construct a TensorFlow model, and the notebook follows these steps: \n\n1. Fetching data from Cloud Storage, \n\n2. Employing TensorFlow Transform for data preprocessing, \n\n3. Utilizing native TensorFlow operators to define a sequential Keras model, \n\n4. Conducting model training and evaluation using model.fit() within the notebook instance, and \n\n5. Storing the trained model in Cloud Storage for serving. \n\nYour objective is to orchestrate a weekly model retraining pipeline with minimal cost, refactoring, and monitoring efforts. How should you proceed to achieve this?",
            "options": [
                "A. Add relevant parameters to the notebook cells and set a recurring run in Vertex AI Workbench.",
                "B. Use TensorFlow Extended (TFX) with Google Cloud executors to define your pipeline, and automate the pipeline to run on Cloud Composer.",
                "C. Use Kubeflow Pipelines SDK with Google Cloud executors to define your pipeline, and use Cloud Scheduler to automate the pipeline to run on Vertex AI Pipelines.",
                "D. Use TensorFlow Extended (TFX) with Google Cloud executors to define your pipeline, and use Cloud Scheduler to automate the pipeline to run on Vertex AI Pipelines."
            ],
            "answer": "C. Use Kubeflow Pipelines SDK with Google Cloud executors to define your pipeline, and use Cloud Scheduler to automate the pipeline to run on Vertex AI Pipelines."
        },
        {
            "question": "You used Vertex AI Workbench user-managed notebooks to develop a TensorFlow model. The model pipeline accesses data from Cloud Storage, performs feature engineering and training locally, and outputs the trained model in Vertex AI Model Registry. The end-to-end pipeline takes 10 hours on the attached optimized instance type. You want to introduce model and data lineage for automated re-training runs for this pipeline only while minimizing the cost to run the pipeline. What should you do?",
            "options": [
                "A. 1. Use the Vertex AI SDK to create an experiment for the pipeline runs, and save metadata throughout the pipeline. 2. Configure a scheduled recurring execution for the notebook. 3. Access data and model metadata in Vertex ML Metadata.",
                "B. 1. Use the Vertex AI SDK to create an experiment, launch a custom training job in Vertex training service with the same instance type configuration as the notebook, and save metadata throughout the pipeline. 2. Configure a scheduled recurring execution for the notebook. 3. Access data and model metadata in Vertex ML Metadata.",
                "C. 1. Refactor the pipeline code into a TensorFlow Extended (TFX) pipeline. 2. Load the TFX pipeline in Vertex AI Pipelines, and configure the pipeline to use the same instance type configuration as the notebook. 3. Use Cloud Scheduler to configure a recurring execution for the pipeline. 4. Access data and model metadata in Vertex AI Pipelines.",
                "D. 1. Create a Cloud Storage bucket to store metadata. 2. Write a function that saves data and model metadata by using TensorFlow ML Metadata in one time-stamped subfolder per pipeline run. 3. Configure a scheduled recurring execution for the notebook. 4. Access data and model metadata in Cloud Storage."
            ],
            "answer": "A. 1. Use the Vertex AI SDK to create an experiment for the pipeline runs, and save metadata throughout the pipeline. 2. Configure a scheduled recurring execution for the notebook. 3. Access data and model metadata in Vertex ML Metadata."
        },
        {
            "question": "You are using AI Platform and TPUs to train a ResNet model for categorizing different defect types in automobile engines. After capturing the training profile with the Cloud TPU profiler plugin, you notice that the process is significantly input-bound. To alleviate this bottleneck and accelerate the training, which two modifications should you consider for the tf.data dataset? (Choose two options)",
            "options": [
                "A. Use the interleave option for reading data.",
                "B. Reduce the value of the repeat parameter.",
                "C. Increase the buffer size for the shuffle option.",
                "D. Set the prefetch option equal to the training batch size.",
                "E. Decrease the batch size argument in your transformation."
            ],
            "answer": [
                "A. Use the interleave option for reading data.",
                "D. Set the prefetch option equal to the training batch size."
            ]
        },
        {
            "question": "You're creating a Kubeflow pipeline on Google Kubernetes Engine, where the initial step involves querying BigQuery. The query results will serve as input for the subsequent step in your pipeline. What is the simplest method to accomplish this?",
            "options": [
                "A. Use the BigQuery console to execute your query, and then save the query results into a new BigQuery table.",
                "B. Write a Python script that uses the BigQuery API to execute queries against BigQuery. Execute this script as the first step in your Kubeflow pipeline.",
                "C. Use the Kubeflow Pipelines domain-specific language to create a custom component that uses the Python BigQuery client library to execute queries.",
                "D. Locate the Kubeflow Pipelines repository on GitHub. Find the BigQuery Query Component, copy that component's URL, and use it to load the component into your pipeline. Use the component to execute queries against BigQuery."
            ],
            "answer": "D. Locate the Kubeflow Pipelines repository on GitHub. Find the BigQuery Query Component, copy that component's URL, and use it to load the component into your pipeline. Use the component to execute queries against BigQuery."
        },
        {
            "question": "You work for an advertising company and aim to evaluate the effectiveness of your latest advertising campaign. You've streamed 500 MB of campaign data into BigQuery and want to query the table, followed by manipulating the query results using a pandas dataframe in an AI Platform notebook. What's the recommended approach?",
            "options": [
                "A. Use AI Platform Notebooks' BigQuery cell magic to query the data, and ingest the results as a pandas dataframe.",
                "B. Export your table as a CSV file from BigQuery to Google Drive, and use the Google Drive API to ingest the file into your notebook instance.",
                "C. Download your table from BigQuery as a local CSV file, and upload it to your AI Platform notebook instance. Use pandas.read_csv to ingest the file as a pandas dataframe.",
                "D. From a bash cell in your AI Platform notebook, use the bq extract command to export the table as a CSV file to Cloud Storage, and then use gsutil cp to copy the data into the notebook. Use pandas.read_csv to ingest the file as a pandas dataframe."
            ],
            "answer": "A. Use AI Platform Notebooks' BigQuery cell magic to query the data, and ingest the results as a pandas dataframe."
        },
        {
            "question": "Your team is in the process of developing a convolutional neural network (CNN)-based architecture from the ground up. Initial experiments conducted on your on-premises CPU-only infrastructure have shown promising results, but the model's convergence is slow. To expedite the model training process and shorten time-to-market, you are considering conducting experiments on Google Cloud virtual machines (VMs) equipped with more powerful hardware. It's important to note that your code doesn't involve manual device placement, and it hasn't been encapsulated within the Estimator model-level abstraction. Given this context, which environment should you choose for training your model?",
            "options": [
                "A. VM on Compute Engine and 1 TPU with all dependencies installed manually.",
                "B. VM on Compute Engine and 8 GPUs with all dependencies installed manually.",
                "C. A Deep Learning VM with an n1-standard-2 machine and 1 GPU with all libraries pre-installed.",
                "D. A Deep Learning VM with more powerful CPU e2-highcpu-16 machines with all libraries pre-installed."
            ],
            "answer": "C. A Deep Learning VM with an n1-standard-2 machine and 1 GPU with all libraries pre-installed."
        },
        {
            "question": "Your team has been assigned the responsibility of developing a machine learning (ML) solution within Google Cloud to categorize support requests for one of your platforms. After analyzing the requirements, you've chosen to utilize TensorFlow for constructing the classifier, providing you with complete control over the model's code, serving, and deployment. Your intention is to leverage Kubeflow pipelines as the ML platform to expedite the process. In order to save time, you aim to build upon existing resources and utilize managed services, rather than starting from scratch. How should you proceed in building the classifier?",
            "options": [
                "A. Use the Natural Language API to classify support requests.",
                "B. Use Vertex AI AutoML Natural Language to build the support requests classifier.",
                "C. Use an established text classification model on AI Platform to perform transfer learning.",
                "D. Use an established text classification model on AI Platform as-is to classify support requests."
            ],
            "answer": "C. Use an established text classification model on AI Platform to perform transfer learning."
        },
        {
            "question": "You're involved in a project that utilizes Neural Networks. The dataset you've been given contains columns with varying ranges. During the data preparation process for model training, you observe that gradient optimization is struggling to converge to a favorable solution. What is the recommended action to take?",
            "options": [
                "A. Use feature construction to combine the strongest features.",
                "B. Use the representation transformation (normalization) technique.",
                "C. Improve the data cleaning step by removing features with missing values.",
                "D. Change the partitioning step to reduce the dimension of the test set and have a larger training set."
            ],
            "answer": "B. Use the representation transformation (normalization) technique."
        }
    ]

    # Function to create a table of contents
    def create_toc(questions):
        toc = ""
        for idx, q in enumerate(questions):
            toc += f"- [Question {idx + 1}](#{'t1-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 1")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}", anchor=f"t1-question-{idx+1}")
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"t1_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

with tab2:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

    # Questions and their options
    questions = [
        {
            "question": "Your organization's call center has tasked you with developing a model to analyze customer sentiments in each call. With over one million calls received daily, the data is stored in Cloud Storage. It's imperative that the data remains within the region of the call's origin and that no Personally Identifiable Information (PII) is stored or analyzed. Additionally, the data science team utilizes a third-party visualization tool that requires an SQL ANSI-2011 compliant interface. Given these requirements, what components should you select for data processing and analytics to effectively design the data pipeline?",
            "options": [
                "A. 1 = Google Cloud Dataflow, 2 = Google BigQuery",
                "B. 1 = Google Cloud Pub/Sub, 2 = Google Cloud Datastore",
                "C. 1 = Google Cloud Dataflow, 2 = Google Cloud SQL",
                "D. 1 = Google Cloud Functions, 2 = Google Cloud SQL"
            ],
            "answer": "A. 1 = Google Cloud Dataflow, 2 = Google BigQuery"
        },
        {
            "question": "You've trained a model that necessitated resource-intensive preprocessing operations on a dataset. Now, these preprocessing steps must be replicated during prediction. Given that the model is deployed on AI Platform for high-throughput online predictions, what is the most suitable architectural approach to use?",
            "options": [
                "A. Validate the accuracy of the model that you trained on preprocessed data. Create a new model that uses the raw data and is available in real time. Deploy the new model onto AI Platform for online prediction.",
                "B. Send incoming prediction requests to a Pub/Sub topic. Transform the incoming data using a Dataflow job. Submit a prediction request to AI Platform using the transformed data. Write the predictions to an outbound Pub/Sub queue.",
                "C. Stream incoming prediction request data into Cloud Spanner. Create a view to abstract your preprocessing logic. Query the view every second for new records. Submit a prediction request to AI Platform using the transformed data. Write the predictions to an outbound Pub/Sub queue.",
                "D. Send incoming prediction requests to a Pub/Sub topic. Set up a Cloud Function that is triggered when messages are published to the Pub/Sub topic. Implement your preprocessing logic in the Cloud Function. Submit a prediction request to AI Platform using the transformed data. Write the predictions to an outbound Pub/Sub queue."
            ],
            "answer": "B. Send incoming prediction requests to a Pub/Sub topic. Transform the incoming data using a Dataflow job. Submit a prediction request to AI Platform using the transformed data. Write the predictions to an outbound Pub/Sub queue."
        },
        {
            "question": "You've developed an ML model using AI Platform and are now moving it into production. The model currently serves a few thousand queries per second but is facing latency issues. Requests are handled by a load balancer, which distributes them across multiple CPU-only Kubeflow pods on Google Kubernetes Engine (GKE).\n\nTo enhance serving latency without altering the underlying infrastructure, what steps should you take?",
            "options": [
            "A. Significantly increase the max_batch_size TensorFlow Serving parameter.",
            "B. Switch to the tensorflow-model-server-universal version of TensorFlow Serving.",
            "C. Significantly increase the max_enqueued_batches TensorFlow Serving parameter.",
            "D. Recompile TensorFlow Serving using the source to support CPU-specific optimizations. Instruct GKE to choose an appropriate baseline minimum CPU platform for serving nodes."
            ],
            "answer": "D. Recompile TensorFlow Serving using the source to support CPU-specific optimizations. Instruct GKE to choose an appropriate baseline minimum CPU platform for serving nodes."
        },
        {
            "question": "You have a fully operational end-to-end ML pipeline that includes hyperparameter tuning of your ML model using AI Platform. However, the hyperparameter tuning process is taking longer than anticipated, causing delays in the downstream processes. You aim to expedite the tuning job without significantly sacrificing its effectiveness.\n\nWhat actions should you consider? (Choose two options)",
            "options": [
            "A. Decrease the number of parallel trials.",
            "B. Decrease the range of floating-point values.",
            "C. Set the early stopping parameter to TRUE.",
            "D. Change the search algorithm from Bayesian search to random search.",
            "E. Decrease the maximum number of trials during subsequent training phases."
            ],
            "answer": [
            "C. Set the early stopping parameter to TRUE.",
            "E. Decrease the maximum number of trials during subsequent training phases."
            ]
        },
        {
            "question": "As the lead ML Engineer for your company, you are responsible for building ML models to digitize scanned customer forms. You have developed a TensorFlow model that converts the scanned images into text and stores them in Cloud Storage. You need to use your ML model on the aggregated data collected at the end of each day with minimal manual intervention.\n\nWhat should you do?",
            "options": [
            "A. Use the batch prediction functionality of AI Platform.",
            "B. Create a serving pipeline in Compute Engine for prediction.",
            "C. Use Cloud Functions for prediction each time a new data point is ingested.",
            "D. Deploy the model on AI Platform and create a version of it for online inference."
            ],
            "answer": "A. Use the batch prediction functionality of AI Platform."
        },
        {
            "question": "You are tasked with developing a strategy to efficiently organize jobs, models, and versions on AI Platform for your team of over 50 data scientists. Which strategy should you opt for to ensure a clean and scalable organization?",
            "options": [
            "A. Set up restrictive IAM permissions on the AI Platform notebooks so that only a single user or group can access a given instance.",
            "B. Separate each data scientist's work into a different project to ensure that the jobs, models, and versions created by each data scientist are accessible only to that user.",
            "C. Use labels to organize resources into descriptive categories. Apply a label to each created resource so that users can filter the results by label when viewing or monitoring the resources.",
            "D. Set up a BigQuery sink for Cloud Logging logs that is appropriately filtered to capture information about AI Platform resource usage. In BigQuery, create a SQL view that maps users to the resources they are using."
            ],
            "answer": "C. Use labels to organize resources into descriptive categories. Apply a label to each created resource so that users can filter the results by label when viewing or monitoring the resources."
        },
        {
            "question": "You're employed at a credit card company and have been assigned the task of developing a custom fraud detection model using Vertex AI AutoML Tables, leveraging historical data. Your primary goal is to enhance the detection of fraudulent transactions while keeping false positives to a minimum.\n\nWhat optimization objective should you select when training the model?",
            "options": [
            "A. An optimization objective that minimizes Log loss",
            "B. An optimization objective that maximizes the Precision at a Recall value of 0.50",
            "C. An optimization objective that maximizes the area under the precision-recall curve (AUC PR) value",
            "D. An optimization objective that maximizes the area under the receiver operating characteristic curve (AUC ROC) value"
            ],
            "answer": "C. An optimization objective that maximizes the area under the precision-recall curve (AUC PR) value"
        },
        {
            "question": "Your data science team is tasked with conducting rapid experiments involving various features, model architectures, and hyperparameters. They need an efficient way to track the accuracy metrics of these experiments and access the metrics programmatically over time.\n\nWhat approach should they take to achieve this while minimizing manual effort?",
            "options": [
            "A. Use Kubeflow Pipelines to execute the experiments. Export the metrics file, and query the results using the Kubeflow Pipelines API.",
            "B. Use AI Platform Training to execute the experiments. Write the accuracy metrics to BigQuery, and query the results using the BigQuery API.",
            "C. Use AI Platform Training to execute the experiments. Write the accuracy metrics to Cloud Monitoring, and query the results using the Monitoring API.",
            "D. Use AI Platform Notebooks to execute the experiments. Collect the results in a shared Google Sheets file, and query the results using the Google Sheets API."
            ],
            "answer": "A. Use Kubeflow Pipelines to execute the experiments. Export the metrics file, and query the results using the Kubeflow Pipelines API."
        },
        {
            "question": "You are implementing transfer learning to train an image classifier, leveraging a pre-trained EfficientNet model. Your training dataset consists of 20,000 images, and your intention is to retrain the model on a daily basis. To keep infrastructure costs to a minimum, what platform components and configuration environment should you employ?",
            "options": [
            "A. A Deep Learning VM with 4 V100 GPUs and local storage.",
            "B. A Deep Learning VM with 4 V100 GPUs and Cloud Storage.",
            "C. A Google Kubernetes Engine cluster with a V100 GPU Node Pool and an NFS Server",
            "D. An AI Platform Training job using a custom scale tier with 4 V100 GPUs and Cloud Storage"
            ],
            "answer": "D. An AI Platform Training job using a custom scale tier with 4 V100 GPUs and Cloud Storage"
        },
        {
            "question": "Your data science team is in the process of training a PyTorch model for image classification, building upon a pre-trained ResNet model. To achieve optimal performance, you now find the need to conduct hyperparameter tuning for various parameters.\n\nWhat steps should you take in this scenario?",
            "options": [
            "A. Convert the model to a Keras model, and run a Keras Tuner job.",
            "B. Run a hyperparameter tuning job on AI Platform using custom containers.",
            "C. Create a Kuberflow Pipelines instance, and run a hyperparameter tuning job on Katib.",
            "D. Convert the model to a TensorFlow model, and run a hyperparameter tuning job on AI Platform."
            ],
            "answer": "B. Run a hyperparameter tuning job on AI Platform using custom containers."
        },
        {
            "question": "You are tasked with creating an ML model for a social media platform to determine whether a user's uploaded profile photo complies with the requirements. The objective is to provide users with feedback regarding the compliance of their pictures.\n\nWhat approach should you take in constructing the model to minimize the risk of incorrectly accepting a non-compliant image?",
            "options": [
            "A. Use Vertex AI AutoML to optimize the model’s recall to minimize false negatives.",
            "B. Use Vertex AI AutoML to optimize the model’s F1 score to balance the accuracy of false positives and false negatives.",
            "C. Use Vertex AI Workbench user-managed notebooks to build a custom model with three times as many examples of pictures that meet the profile photo requirements.",
            "D. Use Vertex AI Workbench user-managed notebooks to build a custom model with three times as many examples of pictures that do not meet the profile photo requirements."
            ],
            "answer": "A. Use Vertex AI AutoML to optimize the model’s recall to minimize false negatives."
        },
        {
            "question": "You are developing a linear model that involves more than 100 input features, all of which have values ranging from -1 to 1. You have a suspicion that many of these features do not provide valuable information for your model. Your objective is to eliminate the non-informative features while preserving the informative ones in their original state.\n\nWhat technique is most suitable for achieving this goal?",
            "options": [
            "A. Use principal component analysis (PCA) to eliminate the least informative features.",
            "B. Use L1 regularization to reduce the coefficients of uninformative features to 0.",
            "C. After building your model, use Shapley values to determine which features are the most informative.",
            "D. Use an iterative dropout technique to identify which features do not degrade the model when removed."
            ],
            "answer": "B. Use L1 regularization to reduce the coefficients of uninformative features to 0."
        },
        {
            "question": "You are employed at a subscription-based company. You have trained an ensemble of tree and neural network models to forecast customer churn, which is the probability that customers will not renew their annual subscriptions. While the average prediction indicates a 15% churn rate, a specific customer is forecasted to have a 70% likelihood of churning. This customer has a product usage history of 30%, resides in New York City, and has been a customer since 1997. Your objective is to elucidate the distinction between the individual prediction of a 70% churn rate and the average prediction. To achieve this, you intend to employ Vertex Explainable AI.\n\nWhat is your recommended course of action?",
            "options": [
            "A. Train local surrogate models to explain individual predictions.",
            "B. Configure sampled Shapley explanations on Vertex Explainable AI.",
            "C. Configure integrated gradient explanations on Vertex Explainable AI.",
            "D. Measure the effect of each feature as the weight of the feature multiplied by the feature value."
            ],
            "answer": "B. Configure sampled Shapley explanations on Vertex Explainable AI."
        },
        {
            "question": "You are tasked with training a natural language model for text classification, specifically on product descriptions. This dataset comprises millions of examples and contains a vocabulary of 100,000 unique words. Your goal is to preprocess the words individually so that they can be effectively input into a recurrent neural network.\n\nWhat steps should you take to achieve this?",
            "options": [
            "A. Create a hot-encoding of words, and feed the encodings into your model.",
            "B. Identify word embeddings from a pre-trained model, and use the embeddings in your model.",
            "C. Sort the words by frequency of occurrence, and use the frequencies as the encodings in your model.",
            "D. Assign a numerical value to each word from 1 to 100,000 and feed the values as inputs in your model."
            ],
            "answer": "B. Identify word embeddings from a pre-trained model, and use the embeddings in your model."
        },
        {
            "question": "Your company operates an application that gathers news articles from various online sources and delivers them to users. You require a recommendation model that can propose articles to readers based on their current reading material, suggesting articles that are similar.\n\nWhich approach should you employ for this task?",
            "options": [
            "A. Create a collaborative filtering system that recommends articles to a user based on the user’s past behavior.",
            "B. Encode all articles into vectors using word2vec, and build a model that returns articles based on vector similarity.",
            "C. Build a logistic regression model for each user that predicts whether an article should be recommended to a user.",
            "D. Manually label a few hundred articles, and then train an SVM classifier based on the manually classified articles that categorizes additional articles into their respective categories."
            ],
            "answer": "B. Encode all articles into vectors using word2vec, and build a model that returns articles based on vector similarity." 
        },
        {
            "question": "You've received a dataset containing sales predictions derived from your company's marketing efforts. This well-structured data is stored in BigQuery and has been meticulously maintained by a team of data analysts. Your task is to create a report that offers insights into the predictive potential of the data. You've been instructed to run various ML models, ranging from basic models to complex multilayered neural networks. You have only a limited amount of time to collect the results of your experiments.\n\nWhich Google Cloud tools should you employ to efficiently and independently accomplish this task?",
            "options": [
            "A. Use BigQuery ML to run several regression models, and analyze their performance.",
            "B. Read the data from BigQuery using Dataproc, and run several models using SparkML.",
            "C. Use Vertex AI Workbench user-managed notebooks with scikit-learn code for a variety of ML algorithms and performance metrics.",
            "D. Train a custom TensorFlow model with Vertex AI, reading the data from BigQuery featuring a variety of ML algorithms."
            ],
            "answer": "A. Use BigQuery ML to run several regression models, and analyze their performance."
        },
        {
            "question": "You are developing a binary classification ML algorithm that aims to identify whether an image of a scanned document contains a company's logo. However, in the dataset, a significant imbalance exists, with 96% of examples not featuring the logo.\n\nTo ensure the highest confidence in your model's performance, which metrics should you prioritize?",
            "options": [
            "A. F-score where recall is weighed more than precision",
            "B. RMSE",
            "C. F1 score",
            "D. F-score where precision is weighed more than recall"
            ],
            "answer": "A. F-score where recall is weighed more than precision"
        },
        {
            "question": "You are a member of the operations team at a global company that oversees a substantial fleet of on-premises servers situated in a handful of data centers worldwide. Your team is tasked with gathering monitoring data from these servers, which includes details on CPU and memory usage. In the event of a server incident, your team is responsible for resolving the issue. However, incident data has not yet been appropriately labeled. Your management has requested the development of a predictive maintenance solution that utilizes VM monitoring data to identify potential failures and subsequently notifies the service desk team.\n\nWhat should be your initial step in this process?",
            "options": [
            "A. Train a time-series model to predict the machines’ performance values. Configure an alert if a machine’s actual performance values significantly differ from the predicted performance values.",
            "B. Implement a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Train a model to predict anomalies based on this labeled dataset.",
            "C. Develop a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Test this heuristic in a production environment.",
            "D. Hire a team of qualified analysts to review and label the machines’ historical performance data. Train a model based on this manually labeled dataset."
            ],
            "answer": "B. Implement a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Train a model to predict anomalies based on this labeled dataset."
        },
        {
            "question": "You've developed a custom ML model using scikit-learn, and you've encountered longer-than-expected training times. To address this issue, you've decided to migrate your model to Vertex AI Training, and you aim to enhance the model's training efficiency.\n\nWhat should be your initial approach to achieve this goal?",
            "options": [
            "A. Migrate your model to TensorFlow, and train it using Vertex AI Training.",
            "B. Train your model in a distributed mode using multiple Compute Engine VMs.",
            "C. Train your model with DLVM images on Vertex AI, and ensure that your code utilizes NumPy and SciPy internal methods whenever possible.",
            "D. Train your model using Vertex AI Training with GPUs."
            ],
            "answer": "C. Train your model with DLVM images on Vertex AI, and ensure that your code utilizes NumPy and SciPy internal methods whenever possible."
        },
        {
            "question": "You have effectively deployed a substantial and intricate TensorFlow model that was trained on tabular data. Your objective is to predict the lifetime value (LTV) field for each subscription, which is stored in the BigQuery table named ```\"subscription.subscriptionPurchase\"``` within the ```\"my-fortune500-company-project\"``` project.\n\nTo ensure that prediction drift is prevented, which refers to significant changes in feature data distribution in production over time, what steps should you take?",
            "options": [
            "A. Implement continuous retraining of the model daily using Vertex AI Pipelines.",
            "B. Add a model monitoring job where 10% of incoming predictions are sampled 24 hours.",
            "C. Add a model monitoring job where 90% of incoming predictions are sampled 24 hours.",
            "D. Add a model monitoring job where 10% of incoming predictions are sampled every hour."
            ],
            "answer": "D. Add a model monitoring job where 10% of incoming predictions are sampled every hour."
        },
        {
            "question": "You are constructing a linear regression model in BigQuery ML to estimate the probability of a customer buying your company's products. The model relies on a city name variable as a significant predictive feature. To facilitate the training and deployment of the model, your data needs to be structured in columns. You aim to prepare the data with minimal coding while retaining the crucial variables.\n\nWhat is the recommended approach?",
            "options": [
            "A. Use TensorFlow to create a categorical variable with a vocabulary list. Create the vocabulary file, and upload it as part of your model to BigQuery ML.",
            "B. Create a new view with BigQuery that does not include a column with city information",
            "C. Use Cloud Data Fusion to assign each city to a region labeled as 1, 2, 3, 4, or 5, and then use that number to represent the city in the model.",
            "D. Use Dataprep to transform the city column using a one-hot encoding method, and make each city a column with binary values."
            ],
            "answer": "D. Use Dataprep to transform the city column using a one-hot encoding method, and make each city a column with binary values."
        },
        {
            "question": "You are conducting experiments with a built-in distributed XGBoost model in Vertex AI Workbench user-managed notebooks. To split your data into training and validation sets, you use the following BigQuery queries:\n\n```\nCREATE OR REPLACE TABLE ‘myproject.mydataset.training‘ AS\n(SELECT * FROM ‘myproject.mydataset.mytable‘ WHERE RAND() <= 0.8);\nCREATE OR REPLACE TABLE ‘myproject.mydataset.validation‘ AS\n(SELECT * FROM ‘myproject.mydataset.mytable‘ WHERE RAND() <= 0.2);\n```\n\nAfter training the model, you achieve an area under the receiver operating characteristic curve (AUC ROC) value of 0.8. However, after deploying the model to production, you observe that the model's performance has dropped to an AUC ROC value of 0.65.\n\nWhat is the most likely problem occurring?",
            "options": [
            "A. There is training-serving skew in your production environment.",
            "B. There is not a sufficient amount of training data.",
            "C. The tables that you created to hold your training and validation records share some records, and you may not be using all the data in your initial table.",
            "D. The RAND() function generated a number that is less than 0.2 in both instances, so every record in the validation table will also be in the training table."
            ],
            "answer": "C. The tables that you created to hold your training and validation records share some records, and you may not be using all the data in your initial table."
        },
        {
            "question": "You require the creation of classification workflows for multiple structured datasets that are currently housed in BigQuery. Since you will need to perform this classification process repeatedly, you aim to execute the following tasks without the need for manual coding: exploratory data analysis, feature selection, model construction, training, hyperparameter tuning, and deployment.\n\nWhat course of action should you take to achieve this?",
            "options": [
            "A. Train a TensorFlow model on Vertex AI.",
            "B. Train a classification Vertex AutoML model.",
            "C. Run a logistic regression job on BigQuery ML.",
            "D. Use scikit-learn in Notebooks with pandas library."
            ],
            "answer": "B. Train a classification Vertex AutoML model."
        },
        {
            "question": "You are in the process of constructing an ML model to forecast stock market trends, considering a broad spectrum of factors. During your data analysis, you observe that certain features exhibit a substantial range.\n\nTo prevent these high-magnitude features from causing overfitting in the model, what action should you take?",
            "options": [
            "A. Standardize the data by transforming it with a logarithmic function.",
            "B. Apply a principal component analysis (PCA) to minimize the effect of any particular feature.",
            "C. Use a binning strategy to replace the magnitude of each feature with the appropriate bin number.",
            "D. Normalize the data by scaling it to have values between 0 and 1."
            ],
            "answer": "D. Normalize the data by scaling it to have values between 0 and 1."
        },
        {
            "question": "You are employed by a company that offers an anti-spam service for detecting and concealing spam content on social media platforms. Currently, your company relies on a list of 200,000 keywords to identify potential spam posts. If a post contains a significant number of these keywords, it's marked as spam. You are considering incorporating machine learning to assist in identifying spam posts for human review.\n\nWhat is the primary benefit of introducing machine learning in this business scenario?",
            "options": [
            "A. Posts can be compared to the keyword list much more quickly.",
            "B. New problematic phrases can be identified in spam posts.",
            "C. A much longer keyword list can be used to flag spam posts.",
            "D. Spam posts can be flagged using far fewer keywords."
            ],
            "answer": "B. New problematic phrases can be identified in spam posts."
        },
        {
            "question": "You are in the process of creating an ML model to predict house prices. During the data preparation, you encounter a crucial predictor variable, which is the distance from the nearest school. However, you notice that this variable frequently has missing values and lacks significant variance. It's important to note that every instance (row) in your dataset holds significance.\n\nHow should you address the issue of missing data in this context?",
            "options": [
            "A. Delete the rows that have missing values.",
            "B. Apply feature crossing with another column that does not have missing values.",
            "C. Predict the missing values using linear regression.",
            "D. Replace the missing values with zeros."
            ],
            "answer": "C. Predict the missing values using linear regression."
        },
        {
            "question": "You're an ML engineer in an agricultural research team, focusing on a crop disease detection tool for identifying leaf rust spots in crop images as an indicator of disease presence and severity. These spots exhibit variability in shape and size and are indicative of disease severity levels. Your objective is to create a highly accurate solution for predicting disease presence and severity.\n\nWhat steps should you take?",
            "options": [
            "A. Create an object detection model that can localize the rust spots.",
            "B. Develop an image segmentation ML model to locate the boundaries of the rust spots.",
            "C. Develop a template matching algorithm using traditional computer vision libraries.",
            "D. Develop an image classification ML model to predict the presence of the disease."
            ],
            "answer": "B. Develop an image segmentation ML model to locate the boundaries of the rust spots."
        },
        {
            "question": "You are currently involved in the development of a system log anomaly detection model for a cybersecurity organization. This model, built with TensorFlow, is intended for real-time prediction. To facilitate this, you're tasked with setting up a Dataflow pipeline for data ingestion via Pub/Sub and subsequent storage of results in BigQuery. Your primary objective is to minimize serving latency.\n\nWhat steps should you take to achieve this goal?",
            "options": [
            "A. Containerize the model prediction logic in Cloud Run, which is invoked by Dataflow.",
            "B. Load the model directly into the Dataflow job as a dependency, and use it for prediction.",
            "C. Deploy the model to a Vertex AI endpoint, and invoke this endpoint in the Dataflow job.",
            "D. Deploy the model in a TFServing container on Google Kubernetes Engine, and invoke it in the Dataflow job."
            ],
            "answer": "B. Load the model directly into the Dataflow job as a dependency, and use it for prediction."
        },
        {
            "question": "As the Director of Data Science at a sizable company, your Data Science team has recently adopted the Kubeflow Pipelines SDK for managing their training pipelines. However, your team has encountered challenges when trying to seamlessly incorporate their custom Python code into the Kubeflow Pipelines SDK environment.\n\nWhat guidance should you provide to expedite the integration of their code with the Kubeflow Pipelines SDK?",
            "options": [
            "A. Use the func_to_container_op function to create custom components from the Python code.",
            "B. Use the predefined components available in the Kubeflow Pipelines SDK to access Dataproc, and run the custom code there.",
            "C. Package the custom Python code into Docker containers, and use the load_component_from_file function to import the containers into the pipeline.",
            "D. Deploy the custom Python code to Cloud Functions, and use Kubeflow Pipelines to trigger the Cloud Function."
            ],
            "answer": "C. Package the custom Python code into Docker containers, and use the load_component_from_file function to import the containers into the pipeline."
        },
        {
            "question": "You work for a magazine publisher and have been tasked with predicting whether customers will cancel their annual subscription. In your exploratory data analysis, you find that 90% of individuals renew their subscription every year, and only 10% of individuals cancel their subscription. After training a NN Classifier, your model predicts those who cancel their subscription with 99% accuracy and predicts those who renew their subscription with 82% accuracy.\n\nHow should you interpret these results?",
            "options": [
            "A. This is not a good result because the model should have a higher accuracy for those who renew their subscription than for those who cancel their subscription.",
            "B. This is not a good result because the model is performing worse than predicting that people will always renew their subscription.",
            "C. This is a good result because predicting those who cancel their subscription is more difficult, since there is less data for this group.",
            "D. This is a good result because the accuracy across both groups is greater than 80%."
            ],
            "answer": "B. This is not a good result because the model is performing worse than predicting that people will always renew their subscription."
        },
        {
            "question": "You have developed an ML model to detect the sentiment of users’ posts on your company's social media page to identify outages or bugs. You are using Dataflow to provide real-time predictions on data ingested from Pub/Sub. You plan to have multiple training iterations for your model and keep the latest two versions live after every run. You want to split the traffic between the versions in an 80:20 ratio, with the newest model getting the majority of the traffic. You want to keep the pipeline as simple as possible, with minimal management required.\n\nWhat should you do?",
            "options": [
            "A. Deploy the models to a Vertex AI endpoint using the traffic-split=0=80, PREVIOUS_MODEL_ID=20 configuration.",
            "B. Wrap the models inside an App Engine application using the --splits PREVIOUS_VERSION=0.2, NEW_VERSION=0.8 configuration",
            "C. Wrap the models inside a Cloud Run container using the REVISION1=20, REVISION2=80 revision configuration.",
            "D. Implement random splitting in Dataflow using beam.Partition() with a partition function calling a Vertex AI endpoint."
            ],
            "answer": "A. Deploy the models to a Vertex AI endpoint using the traffic-split=0=80, PREVIOUS_MODEL_ID=20 configuration."
        },
        {
            "question": "Your data science team needs to rapidly experiment with various features, model architectures, and hyperparameters. They need to track the accuracy metrics for various experiments and use an API to query the metrics over time.\n\nWhat should they use to track and report their experiments while minimizing manual effort?",
            "options": [
            "A. Use Vertex Al Pipelines to execute the experiments. Query the results stored in MetadataStore using the Vertex Al API.",
            "B. Use Vertex Al Training to execute the experiments. Write the accuracy metrics to BigQuery, and query the results using the BigQuery API.",
            "C. Use Vertex Al Training to execute the experiments. Write the accuracy metrics to Cloud Monitoring, and query the results using the Monitoring API.",
            "D. Use Vertex Al Workbench user-managed notebooks to execute the experiments. Collect the results in a shared Google Sheets file, and query the results using the Google Sheets API."
            ],
            "answer": "A. Use Vertex Al Pipelines to execute the experiments. Query the results stored in MetadataStore using the Vertex Al API."
        },
        {
            "question": "You have created an extensive neural network using TensorFlow Keras, and it's anticipated to require several days for training. The model exclusively relies on TensorFlow's native operations and conducts training with high-precision arithmetic. Your objective is to enhance the code to enable distributed training through tf.distribute.Strategy. Additionally, you aim to configure an appropriate virtual machine instance within Compute Engine to reduce the overall training duration.\n\nWhat steps should you take to achieve this?",
            "options": [
            "A. Select an instance with an attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add MirroredStrategy to the code, and create the model in the strategy’s scope with batch size dependent on the number of replicas.",
            "B. Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharing.",
            "C. Create a TPU virtual machine, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs.",
            "D. Create a TPU node, and gradually scale up the machine type until the optimal execution time is reached. Add TPU initialization at the start of the program, define a distributed TPUStrategy, and create the model in the strategy’s scope with batch size and training steps dependent on the number of TPUs."
            ],
            "answer": "B. Create an instance group with one instance with attached GPU, and gradually scale up the machine type until the optimal execution time is reached. Add TF_CONFIG and MultiWorkerMirroredStrategy to the code, create the model in the strategy’s scope, and set up data autosharing."
        },
        {
            "question": "You recently developed a custom ML model that was trained in Vertex AI on a post-processed training dataset stored in BigQuery. You used a Cloud Run container to deploy the prediction service. The service performs feature lookup and pre-processing and sends a prediction request to a model endpoint in Vertex AI. You want to configure a comprehensive monitoring solution for training-serving skew that requires minimal maintenance.\n\nWhat should you do?",
            "options": [
            "A. Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses email to send alerts. When an alert is received, use the console to diagnose the issue.",
            "B. Update the model hosted in Vertex AI to enable request-response logging. Create a Data Studio dashboard that compares training data and logged data for potential training-serving skew and uses email to send a daily scheduled report.",
            "C. Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged.",
            "D. Update the model hosted in Vertex AI to enable request-response logging. Schedule a daily DataFlow Flex job that uses Tensorflow Data Validation to detect training-serving skew and uses Cloud Logging to send alerts. Set up a Cloud Function to initiate model retraining that is triggered when an alert is logged."
            ],
            "answer": "A. Create a Model Monitoring job for the Vertex AI endpoint that uses the training data in BigQuery to perform training-serving skew detection and uses email to send alerts. When an alert is received, use the console to diagnose the issue."
        },
        {
            "question": "You have a dataset that is split into training, validation, and test sets. All the sets have similar distributions. You have sub-selected the most relevant features and trained a neural network in TensorFlow. TensorBoard plots show the training loss oscillating around 0.9, with the validation loss higher than the training loss by 0.3. You want to update the training regime to maximize the convergence of both losses and reduce overfitting.\n\nWhat should you do?",
            "options": [
            "A. Decrease the learning rate to fix the validation loss, and increase the number of training epochs to improve the convergence of both losses.",
            "B. Decrease the learning rate to fix the validation loss, and increase the number and dimension of the layers in the network to improve the convergence of both losses.",
            "C. Introduce L1 regularization to fix the validation loss, and increase the learning rate and the number of training epochs to improve the convergence of both losses.",
            "D. Introduce L2 regularization to fix the validation loss."
            ],
            "answer": "D. Introduce L2 regularization to fix the validation loss."
        },
        {
            "question": "You trained a model for sentiment analysis in TensorFlow Keras, saved it in SavedModel format, and deployed it with Vertex AI Predictions as a custom container. You selected a random sentence from the test set, and used a REST API call to send a prediction request. The service returned the error:\n\n```\n“Could not find matching concrete function to call loaded from the SavedModel. \nGot: Tensor(\"inputs:0\", shape=(None,), dtype=string). Expected: \nTensorSpec(shape=(None, None), dtype=tf.int64, name='inputs')”.\n```\n\nYou want to update the model’s code and fix the error while following Google-recommended best practices. What should you do?",
            "options": [
            "A. Combine all preprocessing steps in a function, and call the function on the string input before requesting the model’s prediction on the processed input.",
            "B. Combine all preprocessing steps in a function, and update the default serving signature to accept a string input wrapped into the preprocessing function call.",
            "C. Create a custom layer that performs all preprocessing steps, and update the Keras model to accept a string input followed by the custom preprocessing layer.",
            "D. Combine all preprocessing steps in a function, and update the Keras model to accept a string input followed by a Lambda layer wrapping the preprocessing function."
            ],
            "answer": "B. Combine all preprocessing steps in a function, and update the default serving signature to accept a string input wrapped into the preprocessing function call."
        },
        {
            "question": "You have previously deployed an ML model into production, and as part of your ongoing maintenance, you collect all the raw requests directed to your model prediction service on a monthly basis. Subsequently, you select a subset of these requests for evaluation by a human labeling service to assess your model's performance. Over the course of a year, you have observed that your model's performance exhibits varying patterns: at times, there is a significant degradation in performance within a month, while in other instances, it takes several months before any noticeable decrease occurs. It's important to note that utilizing the labeling service incurs significant costs, but you also want to avoid substantial performance drops.\n\nIn light of these considerations, you aim to establish an optimal retraining frequency for your model. This approach should enable you to maintain a consistently high level of performance while simultaneously minimizing operational costs.\n\nWhat steps should you take to achieve this?",
            "options": [
            "A. Train an anomaly detection model on the training dataset, and run all incoming requests through this model. If an anomaly is detected, send the most recent serving data to the labeling service.",
            "B. Identify temporal patterns in your model’s performance over the previous year. Based on these patterns, create a schedule for sending serving data to the labeling service for the next year.",
            "C. Compare the cost of the labeling service with the lost revenue due to model performance degradation over the past year. If the lost revenue is greater than the cost of the labeling service, increase the frequency of model retraining; otherwise, decrease the model retraining frequency.",
            "D. Run training-serving skew detection batch jobs every few days to compare the aggregate statistics of the features in the training dataset with recent serving data. If skew is detected, send the most recent serving data to the labeling service."
            ],
            "answer": "D. Run training-serving skew detection batch jobs every few days to compare the aggregate statistics of the features in the training dataset with recent serving data. If skew is detected, send the most recent serving data to the labeling service."
        },
        {
            "question": "You hold the position of a senior ML engineer at a retail firm. Your objective is to establish a centralized method for monitoring and handling ML metadata, allowing your team to conduct reproducible experiments and generate artifacts.\n\nWhich management solution should you suggest to your team?",
            "options": [
            "A. Store your tf.logging data in BigQuery.",
            "B. Manage all relational entities in the Hive Metastore.",
            "C. Store all ML metadata in Google Cloud’s operations suite.",
            "D. Manage your ML workflows with Vertex ML Metadata."
            ],
            "answer": "D. Manage your ML workflows with Vertex ML Metadata."
        },
        {
            "question": "You're employed by a magazine distributor and have the task of constructing a predictive model to forecast which customers will renew their subscriptions for the next year. You've utilized historical data from your organization as your training dataset and developed a TensorFlow model that has been deployed on AI Platform. Now, you must identify the customer attribute that holds the greatest predictive influence for each prediction generated by the model. How should you proceed with this task?",
            "options": [
            "A. Use AI Platform notebooks to perform a Lasso regression analysis on your model, which will eliminate features that do not provide a strong signal.",
            "B. Stream prediction results to BigQuery. Use BigQuery’s CORR(X1, X2) function to calculate the Pearson correlation coefficient between each feature and the target variable.",
            "C. Use the AI Explanations feature on AI Platform. Submit each prediction request with the ‘explain’ keyword to retrieve feature attributions using the sampled Shapley method.",
            "D. Use the What-If tool in Google Cloud to determine how your model will perform when individual features are excluded. Rank the feature importance in order of those that caused the most significant performance drop when removed from the model."
            ],
            "answer": "C. Use the AI Explanations feature on AI Platform. Submit each prediction request with the ‘explain’ keyword to retrieve feature attributions using the sampled Shapley method."
        },
        {
            "question": "You are employed at a bank and tasked with developing a random forest model for fraud detection. The dataset at your disposal contains transactions, with only 1% of them being flagged as fraudulent.\n\nWhat data transformation strategy should you consider to enhance the classifier's performance?",
            "options": [
            "A. Write your data in TFRecords.",
            "B. Z-normalize all the numeric features.",
            "C. Oversample the fraudulent transaction 10 times.",
            "D. Use one-hot encoding on all categorical features."
            ],
            "answer": "C. Oversample the fraudulent transaction 10 times."
        },
        {
            "question": "You are tasked with developing an input pipeline for a machine learning training model, which needs to process images from various sources with minimal latency. Upon discovering that your input data exceeds available memory capacity, how would you construct a dataset in line with Google's recommended best practices?",
            "options": [
            "A. Create a tf.data.Dataset.prefetch transformation.",
            "B. Convert the images to tf.Tensor objects, and then run Dataset.from_tensor_slices().",
            "C. Convert the images to tf.Tensor objects, and then run tf.data.Dataset.from_tensors().",
            "D. Convert the images into TFRecords, store the images in Google Cloud Storage, and then use the tf.data API to read the images for training."
            ],
            "answer": "D. Convert the images into TFRecords, store the images in Google Cloud Storage, and then use the tf.data API to read the images for training."
        },
        {
            "question": "You oversee a team of data scientists who utilize a cloud-based backend system to submit training jobs. Managing this system has become challenging, and you aim to switch to a managed service. The data scientists on your team work with various frameworks, such as Keras, PyTorch, Theano, Scikit-learn, and custom libraries.\n\nWhat would be your recommended course of action?",
            "options": [
            "A. Use the AI Platform custom containers feature to receive training jobs using any framework.",
            "B. Configure Kubeflow to run on Google Kubernetes Engine and receive training jobs through TF Job.",
            "C. Create a library of VM images on Compute Engine, and publish these images on a centralized repository.",
            "D. Set up Slurm workload manager to receive jobs that can be scheduled to run on your cloud infrastructure."
            ],
            "answer": "A. Use the AI Platform custom containers feature to receive training jobs using any framework."
        },
        {
            "question": "As an employee of a public transportation company, your task is to develop a model that estimates delay times across various transportation routes. This model must provide real-time predictions to users via an app. Given the influence of seasonal changes and population growth on data relevance, the model requires monthly retraining.\n\nWhat is the optimal way to set up the end-to-end architecture of this predictive model, adhering to Google's recommended best practices?",
            "options": [
            "A. Configure Kubeflow Pipelines to schedule your multi-step workflow, from training to deploying the model.",
            "B. Utilize a model trained and deployed on BigQuery ML, and trigger retraining using the scheduled query feature in BigQuery.",
            "C. Write a Cloud Functions script that launches training and deployment jobs on AI Platform, triggered by Cloud Scheduler.",
            "D. Employ Cloud Composer to programmatically schedule a Dataflow job that executes the workflow, from training to deploying the model."
            ],
            "answer": "A. Configure Kubeflow Pipelines to schedule your multi-step workflow, from training to deploying the model."
        },
        {
            "question": "You're developing a serverless ML system architecture to augment customer support tickets with relevant metadata before routing them to support agents. The system requires models for predicting ticket priority, estimating resolution time, and conducting sentiment analysis, aiding agents in strategic decision-making during support request processing.\n\nThe tickets are anticipated to be free of domain-specific terminology or jargon. The proposed architecture will follow this sequence:\n\nWhich endpoints should the Enrichment Cloud Functions call?",
            "options": [
            "A. 1 = AI Platform, 2 = AI Platform, 3 = Vertex AI AutoML Vision",
            "B. 1 = AI Platform, 2 = AI Platform, 3 = Vertex AI AutoML Natural Language",
            "C. 1 = AI Platform, 2 = AI Platform, 3 = Cloud Natural Language API",
            "D. 1 = Cloud Natural Language API, 2 = AI Platform, 3 = Cloud Vision API"
            ],
            "answer": "C. 1 = AI Platform, 2 = AI Platform, 3 = Cloud Natural Language API"
        },
        {
            "question": "As you develop models to classify customer support emails, you initially created TensorFlow Estimator models using small datasets on your local system. To enhance performance, you now plan to train these models with larger datasets.\n\nFor a seamless transition of your models from on-premises to Google Cloud, with minimal code refactoring and infrastructure overhead, what approach should you take?",
            "options": [
            "A. Use AI Platform for distributed training.",
            "B. Create a cluster on Dataproc for training.",
            "C. Create a Managed Instance Group with autoscaling.",
            "D. Use Kubeflow Pipelines to train on a Google Kubernetes Engine cluster."
            ],
            "answer": "A. Use AI Platform for distributed training."
        },
        {
            "question": "You trained a model in a Vertex AI Workbench notebook that has good validation RMSE. You defined 20 parameters with the associated search spaces that you plan to use for model tuning. You want to use a tuning approach that maximizes tuning job speed. You also want to optimize cost, reproducibility, model performance, and scalability where possible if they do not affect speed.\n\nWhat should you do?",
            "options": [
            "A. Set up a cell to run a hyperparameter tuning job using Vertex AI Vizier with val_rmse specified as the metric in the study configuration.",
            "B. Using a dedicated Python library such as Hyperopt or Optuna, configure a cell to run a local hyperparameter tuning job with Bayesian optimization.",
            "C. Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the UI to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Grid Search as an algorithm.",
            "D. Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the command line to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Random Search as an algorithm where maximum trial count is equal to parallel trial count."
            ],
            "answer": "D. Refactor the notebook into a parametrized and dockerized Python script, and push it to Container Registry. Use the command line to set up a hyperparameter tuning job in Vertex AI. Use the created image and include Random Search as an algorithm where maximum trial count is equal to parallel trial count."
        },
        {
            "question": "You are employed by an online retail company developing a visual search engine. You have established an end-to-end ML pipeline on Google Cloud to determine whether an image contains your company's product. Anticipating the release of new products soon, you have incorporated a retraining functionality in the pipeline to accommodate new data for your ML models. Additionally, you aim to utilize AI Platform's continuous evaluation service to maintain high accuracy on your test dataset.\n\nWhat steps should you take?",
            "options": [
            "A. Keep the original test dataset unchanged even if newer products are incorporated into retraining.",
            "B. Extend your test dataset with images of the newer products when they are introduced to retraining.",
            "C. Replace your test dataset with images of the newer products when they are introduced to retraining.",
            "D. Update your test dataset with images of the newer products when your evaluation metrics drop below a pre-decided threshold."
            ],
            "answer": "B. Extend your test dataset with images of the newer products when they are introduced to retraining."
        },
        {
            "question": "You are developing a real-time prediction engine that streams files, potentially containing Personally Identifiable Information (PII), to Google Cloud. To scan these files, you plan to use the Cloud Data Loss Prevention (DLP) API.\n\nWhat measures should you implement to guarantee that the PII remains inaccessible to unauthorized persons?",
            "options": [
            "A. Stream all files to Google Cloud, and then write the data to BigQuery. Periodically conduct a bulk scan of the table using the DLP API.",
            "B. Stream all files to Google Cloud, and write batches of the data to BigQuery. While the data is being written to BigQuery, conduct a bulk scan of the data using the DLP API.",
            "C. Create two buckets of data: Sensitive and Non-sensitive. Write all data to the Non-sensitive bucket. Periodically conduct a bulk scan of that bucket using the DLP API, and move the sensitive data to the Sensitive bucket.",
            "D. Create three buckets of data: Quarantine, Sensitive, and Non-sensitive. Write all data to the Quarantine bucket. Periodically conduct a bulk scan of that bucket using the DLP API, and move the data to either the Sensitive or Non-Sensitive bucket."
            ],
            "answer": "D. Create three buckets of data: Quarantine, Sensitive, and Non-sensitive. Write all data to the Quarantine bucket. Periodically conduct a bulk scan of that bucket using the DLP API, and move the data to either the Sensitive or Non-Sensitive bucket."
        },
        {
            "question": "After deploying several versions of an image classification model on AI Platform, you aim to track and compare their performance over time. What approach should you take to effectively perform this comparison?",
            "options": [
            "A. Compare the loss performance for each model on a held-out dataset.",
            "B. Compare the loss performance for each model on the validation data.",
            "C. Compare the receiver operating characteristic (ROC) curve for each model using the What-If Tool.",
            "D. Compare the mean average precision across the models using the Continuous Evaluation feature."
            ],
            "answer": "D. Compare the mean average precision across the models using the Continuous Evaluation feature."
        },
        {
            "question": "You developed a tree model based on an extensive feature set of user behavioral data. The model has been in production for 6 months. New regulations were just introduced that require anonymizing personally identifiable information (PII), which you have identified in your feature set using the Cloud Data Loss Prevention API. You want to update your model pipeline to adhere to the new regulations while minimizing a reduction in model performance.\n\nWhat should you do?",
            "options": [
            "A. Redact the features containing PII data, and train the model from scratch.",
            "B. Mask the features containing PII data, and tune the model from the last checkpoint.",
            "C. Use key-based hashes to tokenize the features containing PII data, and train the model from scratch.",
            "D. Use deterministic encryption to tokenize the features containing PII data, and tune the model from the last checkpoint."
            ],
            "answer": "C. Use key-based hashes to tokenize the features containing PII data, and train the model from scratch."
        }
    ]

    # Function to create a table of contents
    def create_toc(questions):
        toc = ""
        for idx, q in enumerate(questions):
            toc += f"- [Question {idx + 1}](#{'t2-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 2")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}", anchor=f"t2-question-{idx+1}")

        # Image subline
        if idx+1==1:
            st.image('https://img-c.udemycdn.com/redactor/raw/practice_test_question/2023-12-12_17-55-43-754b07e100efaae01d42bdf0472883b7.png')
        elif idx+1==44:
            st.image('https://img-c.udemycdn.com/redactor/raw/practice_test_question/2023-12-12_19-42-08-be3e696e58969b04f09e9c96b216de0a.png')

        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"t2_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

with tab3:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

    # Questions and their options
    questions = [
        {
            "question": "Your organization aims to enhance the efficiency of its internal shuttle service route, which currently stops at all pick-up points across the city every 30 minutes between 7 am and 10 am. The development team has already created an application on Google Kubernetes Engine, requiring users to confirm their presence and shuttle station one day in advance.\n\nHow should you proceed?",
            "options": [
            "A. 1. Build a tree-based regression model that predicts how many passengers will be picked up at each shuttle station. \n\n 2.Dispatch an appropriately sized shuttle and provide the map with the required stops based on the prediction.",
            "B. 1. Build a tree-based classification model that predicts whether the shuttle should pick up passengers at each shuttle station. \n\n2.Dispatch an available shuttle and provide the map with the required stops based on the prediction.",
            "C. 1. Define the optimal route as the shortest route that passes by all shuttle stations with confirmed attendance at the given time under capacity constraints.\n\n2.Dispatch an appropriately sized shuttle and indicate the required stops on the map.",
            "D. 1. Build a reinforcement learning model with tree-based classification models predicting the presence of passengers at shuttle stops as agents, along with a reward function based on a distance-based metric.\n\n2.Dispatch an appropriately sized shuttle and provide the map with the required stops based on the simulated outcome."
            ],
            "answer": "C. 1. Define the optimal route as the shortest route that passes by all shuttle stations with confirmed attendance at the given time under capacity constraints.\n\n2.Dispatch an appropriately sized shuttle and indicate the required stops on the map."
        },
        {
            "question": "To perform multiple classifications on various structured datasets stored in BigQuery, you aim to execute these steps without coding: exploratory data analysis, feature selection, model construction, training, hyperparameter tuning, and deployment.\n\nWhat is the recommended approach?",
            "options": [
            "A. Configure Vertex AI AutoML Tables to perform the classification task.",
            "B. Execute a BigQuery ML task for logistic regression-based classification.",
            "C. Utilize AI Platform Notebooks to execute the classification model using the pandas library.",
            "D. Employ AI Platform for running the classification model job, configured for hyperparameter tuning."
            ],
            "answer": "A. Configure Vertex AI AutoML Tables to perform the classification task."
        },
        {
            "question": "You're tasked with creating a machine learning recommendation model for your company's e-commerce website using Recommendations AI. What is the best approach to develop recommendations that boost revenue while adhering to established best practices?",
            "options": [
            "A. Use the “Other Products You May Like” recommendation type to increase the click-through rate.",
            "B. Use the “Frequently Bought Together” recommendation type to increase the shopping cart size for each order.",
            "C. Import your user events and then your product catalog to make sure you have the highest quality event stream.",
            "D. Because it will take time to collect and record product data, use placeholder values for the product catalog to test the viability of the model."
            ],
            "answer": "B. Use the “Frequently Bought Together” recommendation type to increase the shopping cart size for each order."
        },
        {
            "question": "You have developed and are managing a production system tasked with predicting sales figures. The accuracy of this model is of paramount importance, as it needs to adapt to market fluctuations. Although the model has remained unchanged since its deployment, there has been a consistent decline in its accuracy.\n\nWhat could be the primary reason for this gradual decrease in model accuracy?",
            "options": [
            "A. Poor data quality",
            "B. Lack of model retraining",
            "C. Too few layers in the model for capturing information",
            "D. Incorrect data split ratio during model training, evaluation, validation, and test"
            ],
            "answer": "B. Lack of model retraining"
        },
        {
            "question": "You aim to redesign your ML pipeline for structured data on Google Cloud. Currently, you employ PySpark for large-scale data transformations, but your pipelines take over 12 hours to run. To accelerate development and pipeline execution, you intend to leverage a serverless tool and SQL syntax.\n\nWith your raw data already migrated to Cloud Storage, how should you construct the pipeline on Google Cloud to meet the speed and processing requirements?",
            "options": [
            "A. Use Data Fusion's GUI to construct the transformation pipelines, and then write the data into BigQuery.",
            "B. Convert your PySpark code into SparkSQL queries for data transformation, and subsequently, execute your pipeline on Dataproc to write the data into BigQuery.",
            "C. Ingest your data into Cloud SQL, convert your PySpark commands into SQL queries to transform the data, and then employ federated queries from BigQuery for machine learning.",
            "D. Ingest your data into BigQuery using BigQuery Load, convert your PySpark commands into BigQuery SQL queries for data transformation, and then save the transformations to a new table."
            ],
            "answer": "D. Ingest your data into BigQuery using BigQuery Load, convert your PySpark commands into BigQuery SQL queries for data transformation, and then save the transformations to a new table."
        },
        {
            "question": "As an ML engineer at a major grocery retail chain with stores across various regions, you have been tasked with developing an inventory prediction model. The model will incorporate features such as region, store location, historical demand, and seasonal popularity. You intend for the algorithm to update its learning daily based on new inventory data.\n\nWhich algorithms would be most suitable for constructing this model?",
            "options": [
            "A. Classification",
            "B. Reinforcement Learning",
            "C. Recurrent Neural Networks (RNN)",
            "D. Convolutional Neural Networks (CNN)"
            ],
            "answer": "C. Recurrent Neural Networks (RNN)"
        },
        {
            "question": "You've developed unit tests for a Kubeflow Pipeline, which depend on custom libraries. To automate these unit tests' execution following each new push to the development branch in Cloud Source Repositories, what steps should you take?",
            "options": [
            "A. Write a script that sequentially performs the push to your development branch and executes the unit tests on Cloud Run.",
            "B. Using Cloud Build, set an automated trigger to execute the unit tests when changes are pushed to your development branch.",
            "C. Set up a Cloud Logging sink to a Pub/Sub topic that captures interactions with Cloud Source Repositories. Configure a Pub/Sub trigger for Cloud Run, and execute the unit tests on Cloud Run.",
            "D. Set up a Cloud Logging sink to a Pub/Sub topic that captures interactions with Cloud Source Repositories. Execute the unit tests using a Cloud Function that is triggered when messages are sent to the Pub/Sub topic."
            ],
            "answer": "B. Using Cloud Build, set an automated trigger to execute the unit tests when changes are pushed to your development branch."
        },
        {
            "question": "You recently developed a regression model based on a training dataset that does not contain personally identifiable information (PII) data in compliance with regulatory requirements. Before deploying the model, you perform post-training analysis on multiple data slices and discover that the model is under-predicting for users who are more than 60 years old. You want to remove age bias while maintaining similar training offline performance.\n\nWhat should you do?",
            "options": [
            "A. Perform correlation analysis on the training feature set against the age column, and remove features that are highly correlated with age from the training and evaluation sets.",
            "B. Review the data distribution for each feature against the bucketized age column for the training and evaluation sets, and introduce preprocessing to even irregular feature distributions.",
            "C. Split the training and evaluation sets for users below and above 60 years old, and train one specialized model for each user group.",
            "D. Apply a calibration layer at post-processing that matches the prediction distributions of users below and above 60 years old."
            ],
            "answer": "B. Review the data distribution for each feature against the bucketized age column for the training and evaluation sets, and introduce preprocessing to even irregular feature distributions."
        },
        {
            "question": "The marketing team at your organization has expressed the need to send biweekly scheduled emails to customers who are anticipated to spend above a variable threshold. This marks the marketing team's first foray into machine learning (ML), and you've been assigned the responsibility of overseeing the implementation. To address this requirement, you initiated a new Google Cloud project and leveraged Vertex AI Workbench to craft a solution that involves model training and batch inference using an XGBoost model, utilizing transactional data stored in Cloud Storage.\n\nYour goal is to establish an end-to-end pipeline that seamlessly delivers predictions to the marketing team in a secure manner while also optimizing for cost-efficiency and minimizing the need for extensive code maintenance.\n\nWhat steps should you take to achieve this objective?",
            "options": [
            "A. Create a scheduled pipeline on Vertex AI Pipelines that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.",
            "B. Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, copies the data to BigQuery, uses BigQuery ML to perform training and batch prediction, and outputs a table in BigQuery with customer emails and expected spending.",
            "C. Create a scheduled notebook on Vertex AI Workbench that accesses the data from Cloud Storage, performs training and batch prediction on the managed notebook instance, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending.",
            "D. Create a scheduled pipeline on Cloud Composer that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and sends an email to the marketing team’s Gmail group email with an attachment that contains an encrypted list of all customer emails and expected spending."
            ],
            "answer": "A. Create a scheduled pipeline on Vertex AI Pipelines that accesses the data from Cloud Storage, uses Vertex AI to perform training and batch prediction, and outputs a file in a Cloud Storage bucket that contains a list of all customer emails and expected spending."
        },
        {
            "question": "You are developing a classification model to support predictions for your company’s various products. The dataset you were given for model development has class imbalance You need to minimize false positives and false negatives\n\nWhat evaluation metric should you use to properly train the model?",
            "options": [
            "A. F1 score",
            "B. Recall",
            "C. Accuracy",
            "D. Precision"
            ],
            "answer": "A. F1 score"
        },
        {
            "question": "You are in the process of training a machine learning model utilizing a dataset stored in BigQuery, and this dataset contains numerous values classified as Personally Identifiable Information (PII). Your objective is to decrease the dataset's sensitivity prior to commencing model training, and it's essential to retain all columns in the dataset as they are crucial for your model.\n\nWhat steps should you take in this situation?",
            "options": [
            "A. Using Dataflow, ingest the columns with sensitive data from BigQuery, and then randomize the values in each sensitive column.",
            "B. Use the Cloud Data Loss Prevention (DLP) API to scan for sensitive data, and use Dataflow with the DLP API to encrypt sensitive values with Format Preserving Encryption.",
            "C. Use the Cloud Data Loss Prevention (DLP) API to scan for sensitive data, and use Dataflow to replace all sensitive data by using the encryption algorithm AES-256 with a salt.",
            "D. Before training, use BigQuery to select only the columns that do not contain sensitive data. Create an authorized view of the data so that sensitive values cannot be accessed by unauthorized individuals."
            ],
            "answer": "B. Use the Cloud Data Loss Prevention (DLP) API to scan for sensitive data, and use Dataflow with the DLP API to encrypt sensitive values with Format Preserving Encryption."
        },
        {
            "question": "You have developed a model trained on data stored in Parquet files, which are accessed through a Hive table hosted on Google Cloud. You performed data preprocessing using PySpark and exported it as a CSV file to Cloud Storage. Following preprocessing, you conducted further steps for model training and evaluation. Now, you intend to parameterize the model training process within Kubeflow Pipelines.\n\nWhat steps should you take to achieve this?",
            "options": [
            "A. Remove the data transformation step from your pipeline.",
            "B. Containerize the PySpark transformation step, and add it to your pipeline.",
            "C. Add a ContainerOp to your pipeline that spins a Dataproc cluster, runs a transformation, and then saves the transformed data in Cloud Storage.",
            "D. Deploy Apache Spark at a separate node pool in a Google Kubernetes Engine cluster. Add a ContainerOp to your pipeline that invokes a corresponding transformation job for this Spark instance."
            ],
            "answer": "C. Add a ContainerOp to your pipeline that spins a Dataproc cluster, runs a transformation, and then saves the transformed data in Cloud Storage."
        },
        {
            "question": "You are employed by an online publisher that distributes news articles to a vast audience of over 50 million readers. As part of your responsibilities, you have developed an AI model designed to make content recommendations for the company's weekly newsletter. A recommendation is deemed successful if the recipient opens the recommended article within two days of the newsletter's publication date and spends at least one minute on the page.\n\nTo calculate the success metric, you have access to all the necessary data in BigQuery, which is updated on an hourly basis. Your model has been trained using data spanning eight weeks, with the observation that its performance tends to decline below an acceptable baseline after five weeks. Additionally, the model's training process requires 12 hours to complete. Your primary objective is to ensure that the model consistently performs above the acceptable baseline while optimizing operational costs.\n\nGiven this scenario, what approach should you adopt to monitor the model effectively and determine when it is necessary to initiate retraining?",
            "options": [
            "A. Use Vertex AI Model Monitoring to detect skew of the input features with a sample rate of 100% and a monitoring frequency of two days.",
            "B. Schedule a cron job in Cloud Tasks to retrain the model every week before the newsletter is created.",
            "C. Schedule a weekly query in BigQuery to compute the success metric.",
            "D. Schedule a daily Dataflow job in Cloud Composer to compute the success metric."
            ],
            "answer": "C. Schedule a weekly query in BigQuery to compute the success metric."
        },
        {
            "question": "You've been tasked with operationalizing a proof-of-concept ML model developed with Keras. This model was trained in a Jupyter notebook on a data scientist's local machine, which includes data validation and model analysis cells. Your goal is to automate and orchestrate these notebook steps for weekly retraining, considering an anticipated increase in training data volume.\n\nTo optimize cost-efficiency and leverage managed services, what steps should you take?",
            "options": [
            "A. Move the Jupyter notebook to a Notebooks instance on the largest N2 machine type, and schedule the execution of the steps in the Notebooks instance using Cloud Scheduler.",
            "B. Write the code as a TensorFlow Extended (TFX) pipeline orchestrated with Vertex AI Pipelines. Use standard TFX components for data validation and model analysis, and use Vertex AI Pipelines for model retraining.",
            "C. Rewrite the steps in the Jupyter notebook as an Apache Spark job, and schedule the execution of the job on ephemeral Dataproc clusters using Cloud Scheduler.",
            "D. Extract the steps contained in the Jupyter notebook as Python scripts, wrap each script in an Apache Airflow BashOperator, and run the resulting directed acyclic graph (DAG) in Cloud Composer."
            ],
            "answer": "B. Write the code as a TensorFlow Extended (TFX) pipeline orchestrated with Vertex AI Pipelines. Use standard TFX components for data validation and model analysis, and use Vertex AI Pipelines for model retraining."
        },
        {
            "question": "You are currently in the process of training an object detection model utilizing a Cloud TPU v2, and you've noticed that the training duration is exceeding your initial expectations. To address this issue in a manner that is both cost-effective and efficient, what course of action should you pursue, as indicated by this simplified Cloud TPU profile trace?",
            "options": [
            "A. Move from Cloud TPU v2 to Cloud TPU v3 and increase batch size.",
            "B. Move from Cloud TPU v2 to 8 NVIDIA V100 GPUs and increase batch size.",
            "C. Rewrite your input function to resize and reshape the input images.",
            "D. Rewrite your input function using parallel reads, parallel processing, and prefetch."
            ],
            "answer": "D. Rewrite your input function using parallel reads, parallel processing, and prefetch."
        },
        {
            "question": "You are in the process of creating a TensorFlow model for a financial institution, which aims to predict the influence of consumer spending on global inflation. Given the large dataset and the need for extended training, with regular checkpoints, your organization has emphasized cost minimization.\n\nWhat hardware should you select for this task?",
            "options": [
            "A. A Vertex AI Workbench user-managed notebooks instance running on an n1-standard-16 with 4 NVIDIA P100 GPUs",
            "B. A Vertex AI Workbench user-managed notebooks instance running on an n1-standard-16 with an NVIDIA P100 GPU",
            "C. A Vertex AI Workbench user-managed notebooks instance running on an n1-standard-16 with a non-preemptible v3-8 TPU",
            "D. A Vertex AI Workbench user-managed notebooks instance running on an n1-standard-16 with a preemptible v3-8 TPU"
            ],
            "answer": "D. A Vertex AI Workbench user-managed notebooks instance running on an n1-standard-16 with a preemptible v3-8 TPU"
        },
        {
            "question": "You have received a request to construct a model using a dataset residing in a medium-sized BigQuery table, approximately 10 GB in size. Your objective is to swiftly assess the suitability of this data for model development. You intend to generate a one-time report that encompasses informative data distribution visualizations as well as advanced statistical analyses, which you will share with fellow ML engineers on your team.\n\nTo achieve maximum flexibility in creating your report, what steps should you take?",
            "options": [
            "A. Use Vertex AI Workbench user-managed notebooks to generate the report.",
            "B. Use the Google Data Studio to create the report.",
            "C. Use the output from TensorFlow Data Validation on Dataflow to generate the report.",
            "D. Use Dataprep to create the report."
            ],
            "answer": "A. Use Vertex AI Workbench user-managed notebooks to generate the report."
        },
        {
            "question": "You are a member of the data science team at a multinational beverage company. Your task is to create an ML model for predicting the profitability of a new line of naturally flavored bottled waters in various locations. You have access to historical data containing information such as product types, product sales volumes, expenses, and profits for all regions.\n\nWhat should you select as the input and output variables for your model?",
            "options": [
            "A. Use latitude, longitude, and product type as features. Use profit as model output.",
            "B. Use latitude, longitude, and product type as features. Use revenue and expenses as model outputs.",
            "C. Use product type and the feature cross of latitude with longitude, followed by binning, as features. Use profit as model output.",
            "D. Use product type and the feature cross of latitude with longitude, followed by binning, as features. Use revenue and expenses as model outputs."
            ],
            "answer": "C. Use product type and the feature cross of latitude with longitude, followed by binning, as features. Use profit as model output."
        },
        {
            "question": "You are in the process of building an ML model that analyzes segmented frames extracted from a video feed and generates bounding boxes around specific objects. Your goal is to automate various stages of your training pipeline, which include data ingestion and preprocessing from Cloud Storage, training the object model along with hyperparameter tuning using Vertex AI jobs, and ultimately deploying the model to an endpoint.\n\nTo orchestrate the entire pipeline while minimizing the need for cluster management, which approach should you adopt?",
            "options": [
            "A. Use Kubeflow Pipelines on Google Kubernetes Engine.",
            "B. Use Vertex AI Pipelines with TensorFlow Extended (TFX) SDK.",
            "C. Use Vertex AI Pipelines with Kubeflow Pipelines SDK.",
            "D. Use Cloud Composer for the orchestration."
            ],
            "answer": "C. Use Vertex AI Pipelines with Kubeflow Pipelines SDK."
        },
        {
            "question": "You work as an ML engineer at a travel company, and you've been studying customers' travel behavior for an extended period. During this time, you've deployed models to forecast customers' vacation patterns. You've noticed that customers' vacation choices are influenced by seasonality and holidays, and these seasonal patterns remain consistent across different years. Your goal is to efficiently store and compare model versions and performance metrics across multiple years.\n\nHow should you approach this task?",
            "options": [
            "A. Store the performance statistics in Cloud SQL. Query that database to compare the performance statistics across the model versions.",
            "B. Create versions of your models for each season per year in Vertex AI. Compare the performance statistics across the models in the Evaluate tab of the Vertex AI UI.",
            "C. Store the performance statistics of each pipeline run in Kubeflow under an experiment for each season per year. Compare the results across the experiments in the Kubeflow UI.",
            "D. Store the performance statistics of each version of your models using seasons and years as events in Vertex ML Metadata. Compare the results across the slices."
            ],
            "answer": "D. Store the performance statistics of each version of your models using seasons and years as events in Vertex ML Metadata. Compare the results across the slices."
        },
        {
            "question": "You are in the process of creating an ML model that aims to classify X-ray images to assess the risk of bone fractures. You've already trained a ResNet model on Vertex AI using a TPU as an accelerator, but you're not satisfied with the training time and memory usage. Your goal is to rapidly iterate on the training code with minimal code modifications and without significantly affecting the model's accuracy.\n\nWhat steps should you take to achieve this?",
            "options": [
            "A. Reduce the number of layers in the model architecture.",
            "B. Reduce the global batch size from 1024 to 256.",
            "C. Reduce the dimensions of the images used in the model.",
            "D. Configure your model to use bfloat16 instead of float32."
            ],
            "answer": "D. Configure your model to use bfloat16 instead of float32."
        },
        {
            "question": "You have recently created a deep learning model using Keras and are currently exploring various training strategies. Initially, you trained the model on a single GPU, but the training process proved to be too slow. Subsequently, you attempted to distribute the training across 4 GPUs using `tf.distribute.MirroredStrategy`, but you did not observe a reduction in training time.\n\nWhat steps should you take next?",
            "options": [
            "A. Distribute the dataset with tf.distribute.Strategy.experimental_distribute_dataset",
            "B. Create a custom training loop.",
            "C. Use a TPU with tf.distribute.TPUStrategy.",
            "D. Increase the batch size."
            ],
            "answer": "D. Increase the batch size."
        },
        {
            "question": "As an ML engineer at a bank responsible for developing an ML-based biometric authentication system for the mobile application, you've been tasked with verifying a customer's identity based on their fingerprint. Fingerprints are considered highly sensitive personal information and cannot be downloaded and stored in the bank's databases.\n\nWhat machine learning strategy should you suggest for training and deploying this ML model?",
            "options": [
            "A. Data Loss Prevention API",
            "B. Federated learning",
            "C. MD5 to encrypt data",
            "D. Differential privacy"
            ],
            "answer": "B. Federated learning"
        },
        {
            "question": "You are employed by a toy manufacturer that has witnessed a significant surge in demand. Your task is to create an ML model to expedite the inspection process for product defects, thereby achieving quicker defect detection. There is unreliable Wi-Fi connectivity within the factory, and the company is eager to implement the new ML model promptly.\n\nWhich model should you select for this purpose?",
            "options": [
            "A. Vertex AI AutoML Vision Edge mobile-high-accuracy-1 model",
            "B. Vertex AI AutoML Vision Edge mobile-low-latency-1 model",
            "C. Vertex AI AutoML Vision model",
            "D. Vertex AI AutoML Vision Edge mobile-versatile-1 model"
            ],
            "answer": "B. Vertex AI AutoML Vision Edge mobile-low-latency-1 model"
        },
        {
            "question": "You are employed at a biotech startup focused on experimenting with deep learning ML models inspired by biological organisms. Your team frequently engages in early-stage experiments involving novel ML model architectures and develops custom TensorFlow operations in C++. Training your models involves large datasets and substantial batch sizes, with a typical batch comprising 1024 examples, each approximately 1 MB in size. Furthermore, the average size of a network, including all weights and embeddings, is 20 GB.\n\nIn light of these requirements, which hardware should you select for your models?",
            "options": [
            "A. A cluster with 2 n1-highcpu-64 machines, each with 8 NVIDIA Tesla V100 GPUs (128 GB GPU memory in total), and a n1-highcpu-64 machine with 64 vCPUs and 58 GB RAM",
            "B. A cluster with 2 a2-megagpu-16g machines, each with 16 NVIDIA Tesla A100 GPUs (640 GB GPU memory in total), 96 vCPUs, and 1.4 TB RAM",
            "C. A cluster with an n1-highcpu-64 machine with a v2-8 TPU and 64 GB RAM",
            "D. A cluster with 4 n1-highcpu-96 machines, each with 96 vCPUs and 86 GB RAM"
            ],
            "answer": "B. A cluster with 2 a2-megagpu-16g machines, each with 16 NVIDIA Tesla A100 GPUs (640 GB GPU memory in total), 96 vCPUs, and 1.4 TB RAM"
        },
        {
            "question": "As an ML engineer tasked with developing training pipelines for ML models, your objective is to establish a comprehensive training pipeline for a TensorFlow model. This model will undergo training using a substantial volume of structured data, amounting to several terabytes. To ensure the pipeline's effectiveness, you aim to incorporate data quality checks before training and model quality assessments after training, all while minimizing development efforts and the necessity for infrastructure management.\n\nHow should you go about constructing and orchestrating this training pipeline?",
            "options": [
            "A. Create the pipeline using Kubeflow Pipelines domain-specific language (DSL) and predefined Google Cloud components. Orchestrate the pipeline using Vertex AI Pipelines.",
            "B. Create the pipeline using TensorFlow Extended (TFX) and standard TFX components. Orchestrate the pipeline using Vertex AI Pipelines.",
            "C. Create the pipeline using Kubeflow Pipelines domain-specific language (DSL) and predefined Google Cloud components. Orchestrate the pipeline using Kubeflow Pipelines deployed on Google Kubernetes Engine.",
            "D. Create the pipeline using TensorFlow Extended (TFX) and standard TFX components. Orchestrate the pipeline using Kubeflow Pipelines deployed on Google Kubernetes Engine."
            ],
            "answer": "B. Create the pipeline using TensorFlow Extended (TFX) and standard TFX components. Orchestrate the pipeline using Vertex AI Pipelines."
        },
        {
            "question": "You are part of a data center team responsible for server maintenance. Your management has tasked you with developing a predictive maintenance solution using monitoring data to detect potential server failures. However, the incident data has not been labeled yet.\n\nWhat should be your initial step in this process?",
            "options": [
            "A. Train a time-series model to predict the machines’ performance values. Configure an alert if a machine’s actual performance values significantly differ from the predicted performance values.",
            "B. Develop a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Use this heuristic to monitor server performance in real time.",
            "C. Hire a team of qualified analysts to review and label the machines’ historical performance data. Train a model based on this manually labeled dataset.",
            "D. Develop a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Train a model to predict anomalies based on this labeled dataset."
            ],
            "answer": "D. Develop a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Train a model to predict anomalies based on this labeled dataset."
        },
        {
            "question": "You are a member of the data science team at a manufacturing firm, and you are currently examining the company's extensive historical sales dataset, which consists of hundreds of millions of records. During your exploratory data analysis, you have several tasks to perform, including the calculation of descriptive statistics like mean, median, and mode, conducting intricate statistical hypothesis tests, and generating various feature-related plots over time. Your goal is to leverage as much of the sales data as feasible for your analyses while keeping computational resource usage to a minimum.\n\nHow should you approach this situation?",
            "options": [
            "A. Visualize the time plots in Google Data Studio. Import the dataset into Vertex Al Workbench user-managed notebooks. Use this data to calculate the descriptive statistics and run the statistical analyses.",
            "B. Spin up a Vertex Al Workbench user-managed notebooks instance and import the dataset. Use this data to create statistical and visual analyses.",
            "C. Use BigQuery to calculate the descriptive statistics. Use Vertex Al Workbench user-managed notebooks to visualize the time plots and run the statistical analyses.",
            "D. Use BigQuery to calculate the descriptive statistics, and use Google Data Studio to visualize the time plots. Use Vertex Al Workbench user-managed notebooks to run the statistical analyses."
            ],
            "answer": "D. Use BigQuery to calculate the descriptive statistics, and use Google Data Studio to visualize the time plots. Use Vertex Al Workbench user-managed notebooks to run the statistical analyses."
        },
        {
            "question": "Your organization operates an online message board, and in recent months, there has been a noticeable uptick in the use of toxic language and instances of bullying within the platform. To address this issue, you implemented an automated text classification system designed to identify and flag comments that exhibit toxic or harmful behavior.\n\nHowever, you've received reports from users who believe that benign comments related to their religion are being incorrectly classified as abusive. Upon closer examination, it's become evident that the false positive rate of your classifier is higher for comments that pertain to certain underrepresented religious groups.\n\nGiven that your team is operating on a limited budget and already stretched thin, what steps should you take to remedy this situation?",
            "options": [
            "A. Add synthetic training data where those phrases are used in non-toxic ways.",
            "B. Remove the model and replace it with human moderation.",
            "C. Replace your model with a different text classifier.",
            "D. Raise the threshold for comments to be considered toxic or harmful."
            ],
            "answer": "A. Add synthetic training data where those phrases are used in non-toxic ways."
        },
        {
            "question": "You designed a 5-billion-parameter language model in TensorFlow Keras that used autotuned tf.data to load the data in memory. You created a distributed training job in Vertex AI with `tf.distribute.MirroredStrategy`, and set the large_model_v100 machine for the primary instance. The training job fails with the following error:\n\n```\nThe replica 0 ran out of memory with a non-zero status of 9.\n```\n\nYou want to fix this error without vertically increasing the memory of the replicas.\n\nWhat should you do?",
            "options": [
            "A. Keep MirroredStrategy. Increase the number of attached V100 accelerators until the memory error is resolved.",
            "B. Switch to ParameterServerStrategy, and add a parameter server worker pool with large_model_v100 instance type.",
            "C. Switch to tf.distribute.MultiWorkerMirroredStrategy with Reduction Server. Increase the number of workers until the memory error is resolved.",
            "D. Switch to a custom distribution strategy that uses TF_CONFIG to equally split model layers between workers. Increase the number of workers until the memory error is resolved."
            ],
            "answer": "C. Switch to tf.distribute.MultiWorkerMirroredStrategy with Reduction Server. Increase the number of workers until the memory error is resolved."
        },
        {
            "question": "You hold the role of an ML engineer within a mobile gaming company. A fellow data scientist on your team has recently trained a TensorFlow model, and it falls upon you to integrate this model into a mobile application. However, you've encountered an issue where the current model's inference latency exceeds acceptable production standards. To rectify this, you aim to decrease the inference time by 50%, and you are open to a slight reduction in model accuracy to meet the latency requirement without initiating a new training process.\n\nIn pursuit of this objective, what initial model optimization technique should you consider for latency reduction?",
            "options": [
            "A. Dynamic range quantization",
            "B. Weight pruning",
            "C. Model distillation",
            "D. Dimensionality reduction"
            ],
            "answer": "A. Dynamic range quantization"
        },
        {
            "question": "You are employed by a gaming company that oversees a popular online multiplayer game featuring 5-minute battles between teams of 6 players. With a continuous influx of new players daily, your task is to develop a real-time model for automatically assigning available players to teams. According to user research, battles are more enjoyable when players of similar skill levels are matched together.\n\nWhat key business metrics should you monitor to evaluate the performance of your model?",
            "options": [
            "A. Average time players wait before being assigned to a team",
            "B. Precision and recall of assigning players to teams based on their predicted versus actual ability",
            "C. User engagement as measured by the number of battles played daily per user",
            "D. Rate of return as measured by additional revenue generated minus the cost of developing a new model"
            ],
            "answer": "C. User engagement as measured by the number of battles played daily per user"
        },
        {
            "question": "As an ML engineer working in the contact center of a large enterprise, your task is to develop a sentiment analysis tool for predicting customer sentiment based on recorded phone conversations. You need to determine the optimal approach to build this model while ensuring that factors such as the gender, age, and cultural differences of the customers who contacted the contact center do not influence any stage of the model development pipeline or its outcomes.\n\nWhat steps should you take to address this challenge?",
            "options": [
            "A. Convert the speech to text and extract sentiments based on the sentences.",
            "B. Convert the speech to text and build a model based on the words.",
            "C. Extract sentiment directly from the voice recordings.",
            "D. Convert the speech to text and extract sentiment using syntactical analysis."
            ],
            "answer": "A. Convert the speech to text and extract sentiments based on the sentences."
        },
        {
            "question": "You have a model that is trained using data from a third-party data broker, and you are facing challenges because the data broker does not consistently inform you about formatting changes in the data. You aim to enhance the resilience of your model training pipeline to address such issues.\n\nWhat steps should you take?",
            "options": [
            "A. Use TensorFlow Data Validation to detect and flag schema anomalies.",
            "B. Use TensorFlow Transform to create a preprocessing component that will normalize data to the expected distribution, and replace values that don’t match the schema with 0.",
            "C. Use tf.math to analyze the data, compute summary statistics, and flag statistical anomalies.",
            "D. Use custom TensorFlow functions at the start of your model training to detect and flag known formatting errors."
            ],
            "answer": "A. Use TensorFlow Data Validation to detect and flag schema anomalies."
        },
        {
            "question": "You have recently developed the initial version of an image segmentation model for a self-driving car. Upon deploying the model, you notice a decline in the area under the curve (AUC) metric. Additionally, upon reviewing video recordings, you find that the model performs poorly in densely congested traffic scenarios but functions correctly in lower-traffic situations.\n\nWhat is the most probable explanation for this outcome?",
            "options": [
            "A. The model is overfitting in areas with less traffic and underfitting in areas with more traffic.",
            "B. AUC is not the correct metric to evaluate this classification model.",
            "C. Too much data representing congested areas was used for model training.",
            "D. Gradients become small and vanish while backpropagating from the output to input nodes."
            ],
            "answer": "A. The model is overfitting in areas with less traffic and underfitting in areas with more traffic."
        },
        {
            "question": "You have recently developed a proof-of-concept (POC) deep learning model, and while you are satisfied with the overall architecture, you need to fine-tune a couple of hyperparameters. Specifically, you want to perform hyperparameter tuning on Vertex AI to determine the optimal values for the embedding dimension of a categorical feature and the learning rate. Here are the configurations you have set:\n\n - For the embedding dimension, you have defined the type as INTEGER with a range from a minimum value of 16 to a maximum value of 64.\n - For the learning rate, you have defined the type as DOUBLE with a range from a minimum value of 10e-05 to a maximum value of 10e-02.\n\nYou are utilizing the default Bayesian optimization tuning algorithm, and your primary goal is to maximize the accuracy of the model. Training time is not a significant concern. In this context, how should you configure the hyperparameter scaling for each hyperparameter, and what should be the setting for maxParallelTrials?",
            "options": [
            "A. Use UNIT_LINEAR_SCALE for the embedding dimension, UNIT_LOG_SCALE for the learning rate, and a large number of parallel trials.",
            "B. Use UNIT_LINEAR_SCALE for the embedding dimension, UNIT_LOG_SCALE for the learning rate, and a small number of parallel trials.",
            "C. Use UNIT_LOG_SCALE for the embedding dimension, UNIT_LINEAR_SCALE for the learning rate, and a large number of parallel trials.",
            "D. Use UNIT_LOG_SCALE for the embedding dimension, UNIT_LINEAR_SCALE for the learning rate, and a small number of parallel trials."
            ],
            "answer": "B. Use UNIT_LINEAR_SCALE for the embedding dimension, UNIT_LOG_SCALE for the learning rate, and a small number of parallel trials."
        },
        {
            "question": "While developing an image recognition model using PyTorch with the ResNet50 architecture, your code has been successfully tested on a small subsample using your local laptop. However, your full dataset consists of 200,000 labeled images, and you aim to efficiently scale your training workload while keeping costs low. You have access to 4 V100 GPUs.\n\nWhat steps should you take to achieve this?",
            "options": [
            "A. Create a Google Kubernetes Engine cluster with a node pool that has 4 V100 GPUs. Prepare and submit a TFJob operator to this node pool.",
            "B. Create a Vertex AI Workbench user-managed notebooks instance with 4 V100 GPUs, and use it to train your model.",
            "C. Package your code with Setuptools, and use a pre-built container. Train your model with Vertex AI using a custom tier that contains the required GPUs.",
            "D. Configure a Compute Engine VM with all the dependencies that launches the training. Train your model with Vertex AI using a custom tier that contains the required GPUs."
            ],
            "answer": "C. Package your code with Setuptools, and use a pre-built container. Train your model with Vertex AI using a custom tier that contains the required GPUs."
        },
        {
            "question": "You trained a text classification model. You have the following SignatureDefs:\n\n```\nsignature_def['serving_default']:\n```\n\nThe given SavedModel SignatureDef contains the following input (s):\n\n ```inputs['text'] tensor_info:\n dtype: DT_STRING\n shape: (-1, 2)\n name: serving_default_text: 0\n```\nThe given SavedModel SignatureDef contains the following output (s):\n\n ```\noutputs ['softmax'] tensor_info:\n dtype: DT_FLOAT\n shape: (-1, 2)\n name: StatefulPartitionedCall:0\nMethod name is: tensorflow/serving/predict\n```\n\nYou started a TensorFlow-serving component server and tried to send an HTTP request to get a prediction using:\n\n```\nheaders = {\"content -type\": \"application/json\"}\njson_response = \nrequests.post('http://localhost:8501/v1/models/text_model:predict', data=data, \nheaders=headers)\n```\n\n What is the correct way to write the predict request?",
            "options": [
            "A. `data = json.dumps({\"signature_name\": “seving_default”, “instances” [[‘ab’, ‘bc’, ‘cd’]]})`",
            "B. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’, ‘c’, ‘d’, ‘e’, ‘f’]]})`",
            "C. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’, ‘c’], [‘d’, ‘e’, ‘f’]]})`",
            "D. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’], [‘c’, ‘d’], [‘e’, ‘f’]]})`"
            ],
            "answer": "D. `data = json.dumps({\"signature_name\": “serving_default”, “instances” [[‘a’, ‘b’], [‘c’, ‘d’], [‘e’, ‘f’]]})`"
        },
        {
            "question": "As an employee at a social media company, your task is to identify whether uploaded images feature cars. Each training sample belongs to precisely one category. Having trained an object detection neural network, you've deployed this model version to AI Platform Prediction for evaluation, also setting up an evaluation job linked to this model version. You observe that the model's precision falls short of the required business standards.\n\nWhat adjustments should you make to the softmax threshold in the model's final layer to improve precision?",
            "options": [
            "A. Increase the recall.",
            "B. Decrease the recall.",
            "C. Increase the number of false positives.",
            "D. Decrease the number of false negatives."
            ],
            "answer": "B. Decrease the recall."
        },
        {
            "question": "You are training a computer vision model to identify the type of government ID in images, using a GPU-powered virtual machine on Google Compute Engine. The training parameters include:\n - Optimizer: SGD,\n - Image shape: 224x224,\n - Batch size: 64,\n - Epochs: 10,\n - Verbose: 2.\n\nHowever, you encounter a ```ResourceExhaustedError: Out Of Memory (OOM) when allocating tensor``` during training.\n\nWhat steps should you take to resolve this issue?",
            "options": [
            "A. Change the optimizer.",
            "B. Reduce the batch size.",
            "C. Change the learning rate.",
            "D. Reduce the image shape."
            ],
            "answer": "B. Reduce the batch size."
        },
        {
            "question": "You are tasked with creating a custom deep neural network in Keras to forecast customer purchases based on their purchase history. To assess the performance across various model architectures, while storing training data and comparing evaluation metrics on a unified dashboard, what approach should you adopt?",
            "options": [
            "A. Create multiple models using Vertex AI AutoML Tables.",
            "B. Automate multiple training runs using Cloud Composer.",
            "C. Run multiple training jobs on AI Platform with similar job names.",
            "D. Create an experiment in Kubeflow Pipelines to organize multiple runs."
            ],
            "answer": "D. Create an experiment in Kubeflow Pipelines to organize multiple runs."
        },
        {
            "question": "You have developed a model to forecast daily temperatures. Initially, you randomly divided the data, followed by transforming both the training and test datasets. While the model was trained with hourly-updated temperature data and achieved 97% accuracy in testing, its accuracy plummeted to 66% post-deployment in production.\n\nWhat steps can you take to enhance the accuracy of your model in the production environment?",
            "options": [
            "A. Normalize the data for the training, and test datasets as two separate steps.",
            "B. Split the training and test data based on time rather than a random split to avoid leakage.",
            "C. Add more data to your test set to ensure that you have a fair distribution and sample for testing.",
            "D. Apply data transformations before splitting, and cross-validate to make sure that the transformations are applied to both the training and test sets."
            ],
            "answer": "B. Split the training and test data based on time rather than a random split to avoid leakage."
        },
        {
            "question": "You collaborate with a data engineering team that has developed a pipeline to clean the dataset and store it in a Cloud Storage bucket. You've created an ML model and aim to refresh it as soon as new data becomes available. As part of your CI/CD workflow, you intend to automate the execution of a Kubeflow Pipelines training job on a Google Kubernetes Engine (GKE) cluster.\n\nHow should you design this workflow?",
            "options": [
            "A. Configure your pipeline with Dataflow, which saves the files in Cloud Storage. After the file is saved, start the training job on a GKE cluster.",
            "B. Use App Engine to create a lightweight python client that continuously polls Cloud Storage for new files. As soon as a file arrives, initiate the training job.",
            "C. Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub-triggered Cloud Function to start the training job on a GKE cluster.",
            "D. Use Cloud Scheduler to schedule jobs at a regular interval. For the first step of the job, check the timestamp of objects in your Cloud Storage bucket. If there are no new files since the last run, abort the job."
            ],
            "answer": "C. Configure a Cloud Storage trigger to send a message to a Pub/Sub topic when a new file is available in a storage bucket. Use a Pub/Sub-triggered Cloud Function to start the training job on a GKE cluster."
        },
        {
            "question": "Your team is developing an application for a global bank, expected to be used by millions of customers. As part of this project, you've built a forecasting model that predicts customers' account balances three days into the future. The goal is to use these predictions to implement a new feature that will notify users when their account balance is likely to fall below $25.\n\nHow should you deploy and serve these predictions?",
            "options": [
            "A. 1. Create a Pub/Sub topic for each user. \n\n 2.Deploy a Cloud Function that sends a notification when your model predicts that a user's account balance will drop below the $25 threshold.",
            "B. 1. Create a Pub/Sub topic for each user. \n\n 2.Deploy an application on the App Engine standard environment that sends a notification when your model predicts that a user's account balance will drop below the $25 threshold.",
            "C. 1. Build a notification system on Firebase. \n\n 2.Register each user with a user ID on the Firebase Cloud Messaging server, which sends a notification when the average of all account balance predictions drops below the $25 threshold.",
            "D. 1. Build a notification system on Firebase. \n\n 2.Register each user with a user ID on the Firebase Cloud Messaging server, which sends a notification when your model predicts that a user's account balance will drop below the $25 threshold."
            ],
            "answer": "D. 1. Build a notification system on Firebase. \n\n 2.Register each user with a user ID on the Firebase Cloud Messaging server, which sends a notification when your model predicts that a user's account balance will drop below the $25 threshold."
        },
        {
            "question": "You work for a large technology company aiming to modernize its contact center operations. Your task is to develop a solution for classifying incoming calls by product, enabling faster routing to the appropriate support team. The calls have already been transcribed using the Speech-to-Text API. You aim to minimize data preprocessing and development time.\n\nHow should you proceed to build the model?",
            "options": [
            "A. Use the AI Platform Training built-in algorithms to create a custom model.",
            "B. Use Vertex AI AutoML Natural Language to extract custom entities for classification.",
            "C. Use the Cloud Natural Language API to extract custom entities for classification.",
            "D. Build a custom model to identify the product keywords from the transcribed calls, and then run the keywords through a classification algorithm."
            ],
            "answer": "B. Use Vertex AI AutoML Natural Language to extract custom entities for classification."
        },
        {
            "question": "You work for an online travel agency that also sells advertising placements on its website to other companies. You have been asked to predict the most relevant web banner that a user should see next. Security is important to your company. The model latency requirements are 300ms@p99, the inventory is thousands of web banners, and your exploratory analysis has shown that navigation context is a good predictor. You want to implement the simplest solution.\n\nHow should you configure the prediction pipeline?",
            "options": [
            "A. Embed the client on the website, and then deploy the model on AI Platform Prediction.",
            "B. Embed the client on the website, deploy the gateway on App Engine, and then deploy the model on AI Platform Prediction.",
            "C. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Cloud Bigtable for writing and for reading the user's navigation context, and then deploy the model on AI Platform Prediction.",
            "D. Embed the client on the website, deploy the gateway on App Engine, deploy the database on Memorystore for writing and for reading the user's navigation context, and then deploy the model on Google Kubernetes Engine."
            ],
            "answer": "B. Embed the client on the website, deploy the gateway on App Engine, and then deploy the model on AI Platform Prediction."
        },
        {
            "question": "Your team is currently engaged in an NLP research project aimed at predicting the political affiliations of authors based on the articles they have authored. The training dataset for this project is extensive and structured as follows:\n\n```\nAuthorA:Political Party A\nTextA1: [SentenceA11, SentenceA12, SentenceA13, ...]\nTextA2: [SentenceA21, SentenceA22, SentenceA23, ...]\n…\nAuthorB:Political Party B\nTextB1: [SentenceB11, SentenceB12, SentenceB13, ...]\nTextB2: [SentenceB21, SentenceB22, SentenceB23, ...]\n…\nAuthorC:Political Party B\nTextC1: [SentenceC11, SentenceC12, SentenceC13, ...]\nTextC2: [SentenceC21, SentenceC22, SentenceC23, ...]\n…\nAuthorD:Political Party A\nTextD1: [SentenceD11, SentenceD12, SentenceD13, ...]\nTextD2: [SentenceD21, SentenceD22, SentenceD23, ...]\n…\n…\n```\n\nTo maintain the standard 80%-10%-10% data distribution across the training, testing, and evaluation subsets, you should distribute the training examples as follows:",
            "options": [
            "A. Distribute texts randomly across the train-test-eval subsets:\n\n```\nTrain set: [TextA1, TextB2, ...]\nTest set: [TextA2, TextC1, TextD2, ...]\nEval set: [TextB1, TextC2, TextD1, ...]\n```",
            "B. Distribute authors randomly across the train-test-eval subsets:\n\n```\nTrain set: [TextA1, TextA2, TextD1, TextD2, ...]\nTest set: [TextB1, TextB2, ...]\nEval set: [TextC1, TextC2, ...]\n```",
            "C. Distribute sentences randomly across the train-test-eval subsets:\n\n```\nTrain set: [SentenceA11, SentenceA21, SentenceB11, SentenceB21, SentenceC11, SentenceD21, ...]\nTest set: [SentenceA12, SentenceA22, SentenceB12, SentenceC22, SentenceC12, SentenceD22, ...]\nEval set: [SentenceA13, SentenceA23, SentenceB13, SentenceC23, SentenceC13, SentenceD31, ...]\n```",
            "D. Distribute paragraphs of texts (i.e., chunks of consecutive sentences) across the train-test-eval subsets:\n\n```\nTrain set: [SentenceA11, SentenceA12, SentenceD11, SentenceD12, ...]\nTest set: [SentenceA13, SentenceB13, SentenceB21, SentenceD23, SentenceC12, SentenceD13, ...]\nEval set: [SentenceA11, SentenceA22, SentenceB13, SentenceD22, SentenceC23, SentenceD11, ...]\n```"
            ],
            "answer": "B. Distribute authors randomly across the train-test-eval subsets:\n\n```\nTrain set: [TextA1, TextA2, TextD1, TextD2, ...]\nTest set: [TextB1, TextB2, ...]\nEval set: [TextC1, TextC2, ...]\n```"
        },
        {
            "question": "Your company operates a video sharing platform where users can view and upload videos. You're tasked with developing an ML model to forecast which newly uploaded videos will gain the most popularity, allowing these videos to receive priority placement on your company's website.\n\nHow should you determine the success of the model?",
            "options": [
            "A. The model predicts videos as popular if the user who uploads them has over 10,000 likes.",
            "B. The model predicts 97.5% of the most popular clickbait videos measured by number of clicks.",
            "C. The model predicts 95% of the most popular videos measured by watch time within 30 days of being uploaded.",
            "D. The Pearson correlation coefficient between the log-transformed number of views after 7 days and 30 days after publication is equal to 0."
            ],
            "answer": "C. The model predicts 95% of the most popular videos measured by watch time within 30 days of being uploaded."
        },
        {
            "question": "You are employed by a major retailer and have received a request to categorize your customers based on their buying patterns. The buying records of all customers have been uploaded to BigQuery. You have a hunch that there might be multiple distinct customer segments, but you're uncertain about the exact number and the shared characteristics among them. Your goal is to discover the most efficient approach.\n\nWhat steps should you take?",
            "options": [
            "A. Create a k-means clustering model using BigQuery ML. Allow BigQuery to automatically optimize the number of clusters.",
            "B. Create a new dataset in Dataprep that references your BigQuery table. Use Dataprep to identify similarities within each column.",
            "C. Use the Data Labeling Service to label each customer record in BigQuery. Train a model on your labeled data using AutoML Tables. Review the evaluation metrics to understand whether there is an underlying pattern in the data.",
            "D. Get a list of the customer segments from your company’s Marketing team. Use the Data Labeling Service to label each customer record in BigQuery according to the list. Analyze the distribution of labels in your dataset using Data Studio."
            ],
            "answer": "A. Create a k-means clustering model using BigQuery ML. Allow BigQuery to automatically optimize the number of clusters."
        },
        {
            "question": "You have a task to train a regression model using a dataset stored in BigQuery, consisting of 50,000 records. The dataset contains 20 features, a mix of categorical and numerical, and the target variable can have negative values. Your goal is to achieve high model performance while minimizing both effort and training time.\n\nWhat is the most suitable approach to train this regression model efficiently?",
            "options": [
            "A. Create a custom TensorFlow DNN model",
            "B. Use BQML XGBoost regression to train the model.",
            "C. Use Vertex AI AutoML Tables to train the model without early stopping.",
            "D. Use Vertex AI AutoML Tables to train the model with RMSLE as the optimization objective."
            ],
            "answer": "B. Use BQML XGBoost regression to train the model."
        }
        
    ]

    # Function to create a table of contents
    def create_toc(questions):
        toc = ""
        for idx, q in enumerate(questions):
            toc += f"- [Question {idx + 1}](#{'t3-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 3")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions):
        st.subheader(f"Question {idx + 1}", anchor=f"t3-question-{idx+1}")

        # Image subline
        if idx+1==15:
            st.image('https://img-c.udemycdn.com/redactor/raw/practice_test_question/2023-12-12_22-27-25-a6f72a3778688b34c102adb643066a7b.png')
        # elif idx+1==44:
        #     st.image('https://img-c.udemycdn.com/redactor/raw/practice_test_question/2023-12-12_19-42-08-be3e696e58969b04f09e9c96b216de0a.png')

        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"t3_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")


