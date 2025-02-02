import streamlit as st
import secrets

# Initialize session state for the random question
if "current_question" not in st.session_state:
    st.session_state.current_question = None

tab1, tab2, tab3, tab4, tab5, tab6, tab_ep, tab_rand = st.tabs(['Test 1','Test 2','Test 3','Test 4','Test 5','Test 6','Examprepper','Random Question'])

# Questions and their options
questions_t1 = [
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

    # Questions and their options
questions_t2 = [
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
        "answer": "C. Develop a simple heuristic (e.g., based on z-score) to label the machines’ historical performance data. Test this heuristic in a production environment."
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
        "answer": "B. Compare the loss performance for each model on the validation data."
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
 # Questions and their options
questions_t3 = [
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
questions_t4 = [
    {
    "question": "As an ML engineer at a regulated insurance firm, you've been tasked with creating a model to approve or reject insurance applications. What key considerations should you take into account before developing this model?",
    "options": [
      "A. Redaction, reproducibility, and explainability",
      "B. Traceability, reproducibility, and explainability",
      "C. Federated learning, reproducibility, and explainability",
      "D. Differential privacy, federated learning, and explainability"
    ],
    "answer": "B. Traceability, reproducibility, and explainability"
  },
  {
    "question": "You need to develop an online model prediction service that accesses pre-computed near-real-time features and returns a customer churn probability value. The features are saved in BigQuery and updated hourly using a scheduled query. You want this service to be low latency and scalable and require minimal maintenance.\n\nWhat should you do?",
    "options": [
      "A. 1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use Memorystore to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.",
      "B. 1. Configure a Cloud Function that exports features from BigQuery to Memorystore. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Memorystore and performs inference with an in-memory model.",
      "C. 1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use the online service API from Vertex AI Feature Store to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling.",
      "D. 1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use a custom container on Google Kubernetes Engine to deploy a service that performs feature lookup from Vertex AI Feature Store’s online serving API and performs inference with an in-memory model."
    ],
    "answer": "C. 1. Configure a Cloud Function that exports features from BigQuery to Vertex AI Feature Store. 2. Use the online service API from Vertex AI Feature Store to perform feature lookup. Deploy the model as a custom prediction endpoint in Vertex AI, and enable automatic scaling."
  },
  {
    "question": "While executing a model training pipeline on Vertex AI, it has come to your attention that the evaluation step is encountering an out-of-memory error. Your current setup involves the use of TensorFlow Model Analysis (TFMA) within a standard Evaluator component of the TensorFlow Extended (TFX) pipeline for the evaluation process. Your objective is to address this issue and stabilize the pipeline's performance without compromising the quality of evaluation, all while keeping infrastructure overhead to a minimum.\n\nWhat course of action should you take?",
    "options": [
      "A. Include the flag -runner=DataflowRunner in beam_pipeline_args to run the evaluation step on Dataflow.",
      "B. Move the evaluation step out of your pipeline and run it on custom Compute Engine VMs with sufficient memory.",
      "C. Migrate your pipeline to Kubeflow hosted on Google Kubernetes Engine, and specify the appropriate node parameters for the evaluation step.",
      "D. Add tfma.MetricsSpec() to limit the number of metrics in the evaluation step."
    ],
    "answer": "A. Include the flag -runner=DataflowRunner in beam_pipeline_args to run the evaluation step on Dataflow."
  },
  {
    "question": "You work for a company that manages a ticketing platform for a large chain of cinemas. Customers use a mobile app to search for movies they’re interested in and purchase tickets in the app. Ticket purchase requests are sent to Pub/Sub and are processed with a Dataflow streaming pipeline configured to conduct the following steps:\n\n1. Check for availability of the movie tickets at the selected cinema.\n2. Assign the ticket price and accept payment.\n3. Reserve the tickets at the selected cinema.\n4. Send successful purchases to your database.\n\nEach step in this process has low latency requirements (less than 50 milliseconds). You have developed a logistic regression model with BigQuery ML that predicts whether offering a promo code for free popcorn increases the chance of a ticket purchase, and this prediction should be added to the ticket purchase process. You want to identify the simplest way to deploy this model to production while adding minimal latency.\n\nWhat should you do?",
    "options": [
      "A. Run batch inference with BigQuery ML every five minutes on each new set of tickets issued.",
      "B. Export your model in TensorFlow format, and add a tfx_bsl.public.beam.RunInference step to the Dataflow pipeline.",
      "C. Export your model in TensorFlow format, deploy it on Vertex AI, and query the prediction endpoint from your streaming pipeline.",
      "D. Convert your model with TensorFlow Lite (TFLite), and add it to the mobile app so that the promo code and the incoming request arrive together in Pub/Sub."
    ],
    "answer": "B. Export your model in TensorFlow format, and add a tfx_bsl.public.beam.RunInference step to the Dataflow pipeline."
  },
  {
    "question": "You are employed by a gaming company specializing in massively multiplayer online (MMO) games. You have constructed a TensorFlow model designed to forecast whether players will engage in in-app purchases exceeding $10 within the next two weeks. These predictions are intended to tailor each user's game experience. All user data is stored in BigQuery.\n\nWhat is the most effective approach for deploying your model to strike a balance between cost optimization, user experience enhancement, and ease of management?",
    "options": [
      "A. Import the model into BigQuery ML. Make predictions using batch reading data from BigQuery, and push the data to Cloud SQL",
      "B. Deploy the model to Vertex AI Prediction. Make predictions using batch reading data from Cloud Bigtable, and push the data to Cloud SQL.",
      "C. Embed the model in the mobile application. Make predictions after every in-app purchase event is published in Pub/Sub, and push the data to Cloud SQL.",
      "D. Embed the model in the streaming Dataflow pipeline. Make predictions after every in-app purchase event is published in Pub/Sub, and push the data to Cloud SQL."
    ],
    "answer": "A. Import the model into BigQuery ML. Make predictions using batch reading data from BigQuery, and push the data to Cloud SQL"
  },
  {
    "question": "As a data scientist at an industrial equipment manufacturing company, you are currently working on creating a regression model. This model aims to predict the power consumption in the company's manufacturing plants. The model utilizes sensor data gathered from all of the plants, and these sensors generate tens of millions of records daily. Your objective is to set up a daily training schedule for your model, utilizing all the data collected up to the current date. Additionally, you want the model to scale efficiently with minimal development effort.\n\nWhat steps should you take to achieve this?",
    "options": [
      "A. Train a regression model using Vertex AI AutoML Tables.",
      "B. Develop a custom TensorFlow regression model, and optimize it using Vertex AI Training.",
      "C. Develop a custom scikit-learn regression model, and optimize it using Vertex AI Training.",
      "D. Develop a regression model using BigQuery ML."
    ],
    "answer": "D. Develop a regression model using BigQuery ML."
  },
  {
    "question": "You need to create an architecture for serving asynchronous predictions to detect potential failures in a mission-critical machine part. Your system collects data from various sensors on the machine. The goal is to build a model that can predict a failure occurring within the next N minutes based on the average sensor data over the past 12 hours.\n\nHow should you go about designing this architecture?",
    "options": [
      "A. 1. HTTP requests are sent by the sensors to your ML model, which is deployed as a microservice and exposes a REST API for prediction\n\n2.Your application queries a Vertex AI endpoint where you deployed your model.\n\n3.Responses are received by the caller application as soon as the model produces the prediction.",
      "B. 1. Events are sent by the sensors to Pub/Sub, consumed in real time, and processed by a Dataflow stream processing pipeline.\n\n2.The pipeline invokes the model for prediction and sends the predictions to another Pub/Sub topic.\n\n3.Pub/Sub messages containing predictions are then consumed by a downstream system for monitoring.",
      "C. 1. Export your data to Cloud Storage using Dataflow.\n\n2.Submit a Vertex AI batch prediction job that uses your trained model in Cloud Storage to perform scoring on the preprocessed data.\n\n3.Export the batch prediction job outputs from Cloud Storage and import them into Cloud SQL.",
      "D. 1. Export the data to Cloud Storage using the BigQuery command-line tool\n\n2.Submit a Vertex AI batch prediction job that uses your trained model in Cloud Storage to perform scoring on the preprocessed data.\n\n3.Export the batch prediction job outputs from Cloud Storage and import them into BigQuery."
    ],
    "answer": "B. 1. Events are sent by the sensors to Pub/Sub, consumed in real time, and processed by a Dataflow stream processing pipeline.\n\n2.The pipeline invokes the model for prediction and sends the predictions to another Pub/Sub topic.\n\n3.Pub/Sub messages containing predictions are then consumed by a downstream system for monitoring."
  },
  {
    "question": "Your data science team has requested a system that supports scheduled model retraining, Docker containers, and a service that supports autoscaling and monitoring for online prediction requests.\n\nWhich platform components should you select for building this system?",
    "options": [
      "A. Vertex AI Pipelines and App Engine",
      "B. Vertex AI Pipelines, Vertex AI Prediction, and Vertex AI Model Monitoring",
      "C. Cloud Composer, BigQuery ML, and Vertex AI Prediction",
      "D. Cloud Composer, Vertex AI Training with custom containers, and App Engine"
    ],
    "answer": "B. Vertex AI Pipelines, Vertex AI Prediction, and Vertex AI Model Monitoring"
  },
  {
    "question": "You work for a global footwear retailer and need to predict when an item will go out of stock based on historical inventory data. Customer behavior is highly dynamic, as footwear demand is influenced by various factors. Your goal is to train models on all available data but assess their performance on specific data subsets before deploying them to production.\n\nWhat is the most efficient and dependable way to carry out this validation process?",
    "options": [
      "A. Use then TFX ModelValidator tools to specify performance metrics for production readiness.",
      "B. Use k-fold cross-validation as a validation strategy to ensure that your model is ready for production.",
      "C. Use the last relevant week of data as a validation set to ensure that your model is performing accurately on current data.",
      "D. Use the entire dataset and treat the area under the receiver operating characteristics curve (AUC ROC) as the main metric."
    ],
    "answer": "C. Use the last relevant week of data as a validation set to ensure that your model is performing accurately on current data."
  },
  {
    "question": "You possess a substantial collection of written support cases that fall into three distinct categories: Technical Support, Billing Support, or Other Issues. The task at hand is to efficiently construct, evaluate, and implement a system capable of automatically categorizing upcoming written requests into one of these predefined categories.\n\nHow should you structure the pipeline to achieve this?",
    "options": [
      "A. Use the Cloud Natural Language API to obtain metadata to classify the incoming cases.",
      "B. Use Vertex AI AutoML Natural Language to build and test a classifier. Deploy the model as a REST API.",
      "C. Use BigQuery ML to build and test a logistic regression model to classify incoming requests. Use BigQuery ML to perform inference.",
      "D. Create a TensorFlow model using Google’s BERT pre-trained model. Build and test a classifier, and deploy the model using Vertex AI."
    ],
    "answer": "B. Use Vertex AI AutoML Natural Language to build and test a classifier. Deploy the model as a REST API."
  },
  {
    "question": "You recently joined an enterprise-scale company that has thousands of datasets. You know that there are accurate descriptions for each table in BigQuery, and you are searching for the proper BigQuery table to use for a model you are building on AI Platform.\n\nHow should you find the data that you need?",
    "options": [
      "A. Use Data Catalog to search the BigQuery datasets by using keywords in the table description.",
      "B. Tag each of your model and version resources on AI Platform with the name of the BigQuery table that was used for training.",
      "C. Maintain a lookup table in BigQuery that maps the table descriptions to the table ID. Query the lookup table to find the correct table ID for the data that you need.",
      "D. Execute a query in BigQuery to retrieve all the existing table names in your project using the INFORMATION_SCHEMA metadata tables that are native to BigQuery. Use the result to find the table that you need."
    ],
    "answer": "A. Use Data Catalog to search the BigQuery datasets by using keywords in the table description."
  },
  {
    "question": "While performing an exploratory analysis of a dataset, you come across a categorical feature A that exhibits significant predictive power. However, you notice that this feature is sometimes missing values.\n\nWhat course of action should you take?",
    "options": [
      "A. Drop feature A if more than 15% of values are missing. Otherwise, use feature A as-is.",
      "B. Compute the mode of feature A and then use it to replace the missing values in feature A.",
      "C. Replace the missing values with the values of the feature with the highest Pearson correlation with feature A.",
      "D. Add an additional class to categorical feature A for missing values. Create a new binary feature that indicates whether feature A is missing."
    ],
    "answer": "D. Add an additional class to categorical feature A for missing values. Create a new binary feature that indicates whether feature A is missing."
  },
  {
    "question": "As you keep an eye on your model training and observe the GPU utilization, you come to realize that you're using a native synchronous implementation. Moreover, your training data is divided into several files, and you're eager to minimize the execution time of your input pipeline.\n\nWhat steps should you take to address this situation?",
    "options": [
      "A. Increase the CPU load",
      "B. Add caching to the pipeline",
      "C. Increase the network bandwidth",
      "D. Add parallel interleave to the pipeline"
    ],
    "answer": "D. Add parallel interleave to the pipeline"
  },
  {
    "question": "You need to execute a batch prediction on 100 million records in a BigQuery table with a custom TensorFlow DNN regressor model, and then store the predicted results in a BigQuery table. You want to minimize the effort required to build this inference pipeline. What steps should you take to achieve this?",
    "options": [
      "A. Import the TensorFlow model with BigQuery ML, and run the ml.predict function.",
      "B. Use the TensorFlow BigQuery reader to load the data, and use the BigQuery API to write the results to BigQuery.",
      "C. Create a Dataflow pipeline to convert the data in BigQuery to TFRecords. Run a batch inference on Vertex AI Prediction, and write the results to BigQuery.",
      "D. Load the TensorFlow SavedModel in a Dataflow pipeline. Use the BigQuery I/O connector with a custom function to perform the inference within the pipeline, and write the results to BigQuery."
    ],
    "answer": "A. Import the TensorFlow model with BigQuery ML, and run the ml.predict function."
  },
  {
    "question": "You are in the process of building a deep neural network classification model, and your dataset includes categorical input features. Some of these categorical columns have a high cardinality, with over 10,000 unique values.\n\nHow should you handle the encoding of these categorical values for input into the model?",
    "options": [
      "A. Convert each categorical value into an integer value.",
      "B. Convert the categorical string data to one-hot hash buckets.",
      "C. Map the categorical variables into a vector of boolean values.",
      "D. Convert each categorical value into a run-length encoded string."
    ],
    "answer": "B. Convert the categorical string data to one-hot hash buckets."
  },
  {
    "question": "You require a rapid solution for constructing and training a model that can predict the sentiment of customer reviews, utilizing custom categories, all without the need for manual coding. However, your dataset is insufficient to train a model entirely from the ground up. The primary objective is to achieve a high level of predictive accuracy.\n\nIn light of these considerations, which service should you opt for?",
    "options": [
      "A. Vertex AI AutoML Natural Language",
      "B. Cloud Natural Language API",
      "C. AI Hub pre-made Jupyter Notebooks",
      "D. AI Platform Training built-in algorithms"
    ],
    "answer": "A. Vertex AI AutoML Natural Language"
  },
  {
    "question": "You are employed by a small company that has implemented an ML model using autoscaling on Vertex AI to provide online predictions within a production setting. Currently, the model handles approximately 20 prediction requests per hour, with an average response time of one second. However, you've recently retrained the same model using fresh data and are now conducting a canary test by directing approximately 10% of the production traffic to this updated model.\n\nDuring this canary test, you've observed that prediction requests for the new model are taking anywhere from 30 to 180 seconds to finish.\n\nWhat step should you take to address this issue?",
    "options": [
      "A. Submit a request to raise your project quota to ensure that multiple prediction services can run concurrently.",
      "B. Turn off auto-scaling for the online prediction service of your new model. Use manual scaling with one node always available.",
      "C. Remove your new model from the production environment. Compare the new model and existing model codes to identify the cause of the performance bottleneck.",
      "D. Remove your new model from the production environment. For a short trial period, send all incoming prediction requests to BigQuery. Request batch predictions from your new model, and then use the Data Labeling Service to validate your model’s performance before promoting it to production."
    ],
    "answer": "C. Remove your new model from the production environment. Compare the new model and existing model codes to identify the cause of the performance bottleneck."
  },
  {
    "question": "You've created a custom model using Vertex AI to predict your company's product sales, relying on historical transactional data. You foresee potential shifts in feature distributions and correlations between these features in the near future. Additionally, you anticipate a significant influx of prediction requests. In light of this, you intend to employ Vertex AI Model Monitoring for drift detection while keeping costs to a minimum.\n\nWhat step should you take to achieve this?",
    "options": [
      "A. Use the features for monitoring. Set a monitoring-frequency value that is higher than the default.",
      "B. Use the features for monitoring. Set a prediction-sampling-rate value that is closer to 1 than 0.",
      "C. Use the features and the feature attributions for monitoring. Set a monitoring-frequency value that is lower than the default.",
      "D. Use the features and the feature attributions for monitoring. Set a prediction-sampling-rate value that is closer to 0 than 1."
    ],
    "answer": "D. Use the features and the feature attributions for monitoring. Set a prediction-sampling-rate value that is closer to 0 than 1."
  },
  {
    "question": "You have recently deployed a model to a Vertex AI endpoint, and you are encountering frequent data drift. To address this, you have enabled request-response logging and established a Vertex AI Model Monitoring job. However, you've noticed that your model is receiving higher traffic than initially anticipated. Your goal is to reduce the cost of model monitoring while still maintaining the ability to promptly detect drift.\n\nWhat step should you take?",
    "options": [
      "A. Replace the monitoring job with a DataFlow pipeline that utilizes TensorFlow Data Validation (TFDV).",
      "B. Replace the monitoring job with a custom SQL script designed to calculate statistics on the features and predictions within BigQuery.",
      "C. Decrease the sample_rate parameter in the RandomSampleConfig of the monitoring job.",
      "D. Increase the monitor_interval parameter in the ScheduleConfig of the monitoring job."
    ],
    "answer": "C. Decrease the sample_rate parameter in the RandomSampleConfig of the monitoring job."
  },
  {
    "question": "You have recently used TensorFlow to train a classification model on tabular data. You have created a Dataflow pipeline that can transform several terabytes of data into training or prediction datasets consisting of TFRecords. You now need to productionize the model, and you want the predictions to be automatically uploaded to a BigQuery table on a weekly schedule.\n\nWhat should you do?",
    "options": [
      "A. Import the model into Vertex AI and deploy it to a Vertex AI endpoint. On Vertex AI Pipelines, create a pipeline that uses the DataflowPythonJobOp and the ModelBatchPredictOp components.",
      "B. Import the model into Vertex AI and deploy it to a Vertex AI endpoint. Create a Dataflow pipeline that reuses the data processing logic, sends requests to the endpoint, and then uploads predictions to a BigQuery table.",
      "C. Import the model into Vertex AI. On Vertex AI Pipelines, create a pipeline that uses the DataflowPythonJobOp and the ModelBatchPredictOp components.",
      "D. Import the model into BigQuery. Implement the data processing logic in a SQL query. On Vertex AI Pipelines, create a pipeline that uses the BigqueryQueryJobOp and the BigqueryPredictModelJobOp components."
    ],
    "answer": "C. Import the model into Vertex AI. On Vertex AI Pipelines, create a pipeline that uses the DataflowPythonJobOp and the ModelBatchPredictOp components."
  },
  {
    "question": "You have deployed a scikit-learn model to a Vertex AI endpoint using a custom model server. You enabled autoscaling; however, the deployed model fails to scale beyond one replica, leading to dropped requests. You notice that CPU utilization remains low even during periods of high load.\n\nWhat should you do?",
    "options": [
      "A. Attach a GPU to the prediction nodes.",
      "B. Increase the number of workers in your model server.",
      "C. Schedule scaling of the nodes to match expected demand.",
      "D. Increase the minReplicaCount in your DeployedModel configuration."
    ],
    "answer": "B. Increase the number of workers in your model server."
  },
  {
    "question": "Your work for a textile manufacturing company. Your company has hundreds of machines, and each machine has many sensors. Your team used the sensory data to build hundreds of ML models that detect machine anomalies. Models are retrained daily, and you need to deploy these models in a cost-effective way. The models must operate 24/7 without downtime and make sub-millisecond predictions.\n\nWhat should you do?",
    "options": [
      "A. Deploy a Dataflow batch pipeline and a Vertex AI Prediction endpoint.",
      "B. Deploy a Dataflow batch pipeline with the RunInference API, and use model refresh.",
      "C. Deploy a Dataflow streaming pipeline and a Vertex AI Prediction endpoint with autoscaling.",
      "D. Deploy a Dataflow streaming pipeline with the RunInference API, and use automatic model refresh."
    ],
    "answer": "D. Deploy a Dataflow streaming pipeline with the RunInference API, and use automatic model refresh."
  },
  {
    "question": "You are pre-training a large language model on Google Cloud, which involves custom TensorFlow operations in the training loop, a large batch size, and several weeks of training. You aim to configure a training architecture that minimizes both training time and compute costs.\n\nWhat should you do?",
    "options": [
      "A. Implement a TPU Pod slice with -accelerator-type=v4-l28 using `tf.distribute.TPUStrategy`.",
      "B. Implement 8 workers of a2-megagpu-16g machines using `tf.distribute.MultiWorkerMirroredStrategy`.",
      "C. Implement 16 workers of c2d-highcpu-32 machines using ``tf.distribute.MirroredStrategy``.",
      "D. Implement 16 workers of a2-highgpu-8g machines using `tf.distribute.MultiWorkerMirroredStrategy`."
    ],
    "answer": "B. Implement 8 workers of a2-megagpu-16g machines using `tf.distribute.MultiWorkerMirroredStrategy`."
  },
  {
    "question": "You are tasked with the deployment of a scikit-learn classification model into a production environment. This model must be capable of continuously serving requests around the clock, and you anticipate a high volume of requests, possibly reaching millions per second, during the operational hours from 8 am to 7 pm. Your primary objective is to keep deployment costs to a minimum.\n\nHow should you proceed to achieve this?",
    "options": [
      "A. Deploy an online Vertex AI prediction endpoint. Set the max replica count to 1",
      "B. Deploy an online Vertex AI prediction endpoint. Set the max replica count to 100",
      "C. Deploy an online Vertex AI prediction endpoint with one GPU per replica. Set the max replica count to 1",
      "D. Deploy an online Vertex AI prediction endpoint with one GPU per replica. Set the max replica count to 100"
    ],
    "answer": "B. Deploy an online Vertex AI prediction endpoint. Set the max replica count to 100"
  },
  {
    "question": "Your company maintains a substantial collection of audio files from phone calls to your customer call center, stored in an on-premises database. These audio files are in wav format and have an approximate duration of 5 minutes each. Your objective is to analyze these audio files for customer sentiment, and you plan to utilize the Speech-to-Text API. Your goal is to employ the most efficient approach.\n\nWhat steps should you take?",
    "options": [
      "A. 1. Upload the audio files to Cloud Storage.\n\n2.Call the `speech:longrunningrecognize` API endpoint to generate transcriptions.\n\n3.Create a Cloud Function that calls the Natural Language API using the analyzeSentiment method.",
      "B. 1. Upload the audio files to Cloud Storage.\n\n2.Call the `speech:longrunningrecognize` API endpoint to generate transcriptions.\n\n3.Call the predict method of an Vertex AI AutoML sentiment analysis model to analyze the transcriptions.",
      "C. 1. Iterate over your local files in Python.\n\n2.Utilize the Speech-to-Text Python library to create a speech.RecognitionAudio object, setting the content to the audio file data.\n\n3.Call the `speech:recognize` API endpoint to generate transcriptions.\n\n4.Call the predict method of an Vertex AI AutoML sentiment analysis model to analyze the transcriptions.",
      "D. 1. Iterate over your local files in Python.\n\n2.Use the Speech-to-Text Python Library to create a speech.RecognitionAudio object, setting the content to the audio file data.\n\n3.Call the `speech:longrunningrecognize` API endpoint to generate transcriptions.\n\n4.Call the Natural Language API using the analyzeSentiment method."
    ],
    "answer": "A. 1. Upload the audio files to Cloud Storage.\n\n2.Call the `speech:longrunningrecognize` API endpoint to generate transcriptions.\n\n3.Create a Cloud Function that calls the Natural Language API using the analyzeSentiment method."
  },
  {
    "question": "You've created a Vertex AI ML pipeline that involves preprocessing and training stages, and each of these stages operates within distinct custom Docker images. Within your organization, GitHub and GitHub Actions are employed for continuous integration and continuous deployment (CI/CD) to perform unit and integration tests.\n\nTo automate the model retraining process, you seek a workflow that can be triggered manually and automatically whenever new code is merged into the main branch. Your goal is to streamline the workflow while maintaining flexibility. How should you set up and configure the CI/CD workflow to achieve this?",
    "options": [
      "A. Trigger a Cloud Build workflow to run tests, build custom Docker images, push the images to Artifact Registry, and launch the pipeline in Vertex AI Pipelines.",
      "B. Trigger GitHub Actions to run the tests, launch a job on Cloud Run to build custom Docker images, push the images to Artifact Registry, and launch the pipeline in Vertex AI Pipelines.",
      "C. Trigger GitHub Actions to run the tests, build custom Docker images, push the images to Artifact Registry, and launch the pipeline in Vertex AI Pipelines.",
      "D. Trigger GitHub Actions to run the tests, launch a Cloud Build workflow to build custom Docker images, push the images to Artifact Registry, and launch the pipeline in Vertex AI Pipelines."
    ],
    "answer": "C. Trigger GitHub Actions to run the tests, build custom Docker images, push the images to Artifact Registry, and launch the pipeline in Vertex AI Pipelines."
  },
  {
    "question": "You work with a team of researchers to develop state-of-the-art algorithms for financial analysis. Your team develops and debugs complex models in TensorFlow. You want to maintain the ease of debugging while also reducing the model training time.\n\nHow should you set up your training environment?",
    "options": [
      "A. Configure a v3-8 TPU VM. SSH into the VM to train and debug the model.",
      "B. Configure a v3-8 TPU node. Use Cloud Shell to SSH into the Host VM to train and debug the model.",
      "C. Configure a n1 -standard-4 VM with 4 NVIDIA P100 GPUs. SSH into the VM and use ParameterServerStraregv to train the model.",
      "D. Configure a n1-standard-4 VM with 4 NVIDIA P100 GPUs. SSH into the VM and use MultiWorkerMirroredStrategy to train the model."
    ],
    "answer": "D. Configure a n1-standard-4 VM with 4 NVIDIA P100 GPUs. SSH into the VM and use MultiWorkerMirroredStrategy to train the model."
  },
  {
    "question": "You are employed at a retail company and have access to a managed tabular dataset within Vertex AI, which encompasses sales data from three distinct stores. This dataset incorporates various features, including store names and sale timestamps. Your objective is to leverage this data to train a model capable of making sales predictions for an upcoming new store. To accomplish this, you must divide the data into training, validation, and test sets.\n\nWhat approach should you employ for this data split?",
    "options": [
      "A. Use Vertex AI manual split, using the store name feature to assign one store for each set",
      "B. Use Vertex AI default data split",
      "C. Use Vertex AI chronological split, and specify the sales timestamp feature as the time variable",
      "D. Use Vertex AI random split, assigning 70% of the rows to the training set, 10% to the validation set, and 20% to the test set"
    ],
    "answer": "A. Use Vertex AI manual split, using the store name feature to assign one store for each set"
  },
  {
    "question": "You are employed at a bank, and you've developed a customized model to determine whether a loan application should be flagged for human review. The input features required for this model are stored within a BigQuery table. The model has exhibited strong performance, and you are in the process of preparing it for deployment in a production setting. However, due to compliance requirements, it is now imperative that the model provides explanations for each prediction it makes. Your objective is to incorporate this explanatory capability into your model's code with minimal effort while ensuring that the explanations offered are as accurate as possible. How should you proceed to accomplish this?",
    "options": [
      "A. Create an Vertex AI AutoML tabular model by using the BigQuery data with integrated Vertex Explainable AI.",
      "B. Create a BigQuery ML deep neural network model and use the ML.EXPLAIN_PREDICT method with the num_integral_steps parameter.",
      "C. Upload the custom model to Vertex AI Model Registry and configure feature-based attribution by using sampled Shapley with input baselines.",
      "D. Update the custom serving container to include sampled Shapley-based explanations in the prediction outputs."
    ],
    "answer": "C. Upload the custom model to Vertex AI Model Registry and configure feature-based attribution by using sampled Shapley with input baselines."
  },
  {
    "question": "You are employed by a startup that manages various data science workloads. Currently, your compute infrastructure is on-premises, and the data science workloads rely on PySpark. Your team is planning to migrate these data science workloads to Google Cloud. To initiate a proof of concept for migrating one data science job to Google Cloud while minimizing cost and effort, what should be your initial step?",
    "options": [
      "A. Create a n2-standard-4 VM instance and install Java, Scala, and Apache Spark dependencies on it.",
      "B. Create a Google Kubernetes Engine cluster with a basic node pool configuration, install Java, Scala, and Apache Spark dependencies on it.",
      "C. Create a Standard (1 master, 3 workers) Dataproc cluster, and run a Vertex AI Workbench notebook instance on it.",
      "D. Create a Vertex AI Workbench notebook with instance type n2-standard-4."
    ],
    "answer": "C. Create a Standard (1 master, 3 workers) Dataproc cluster, and run a Vertex AI Workbench notebook instance on it."
  },
  {
    "question": "You work for a manufacturing company and your task is to train a custom image classification model to detect product defects at the end of an assembly line. Although your model is performing well, some images in your holdout set are consistently mislabeled with high confidence. You want to use Vertex AI to gain insights into your model’s results.\n\nWhat should you do?",
    "options": [
      "A. Configure feature-based explanations by using Integrated Gradients. Set the visualization type to PIXELS, and set the clip_percent_upperbound to 95.",
      "B. Create an index by using Vertex AI Matching Engine. Query the index with your mislabeled images.",
      "C. Configure feature-based explanations by using XRAI. Set the visualization type to OUTLINES, and set the polarity to positive.",
      "D. Configure example-based explanations. Specify the embedding output layer to be used for the latent space representation."
    ],
    "answer": "D. Configure example-based explanations. Specify the embedding output layer to be used for the latent space representation."
  },
  {
    "question": "You are using Vertex AI and TensorFlow to develop a custom image classification model. You need the model’s decisions and the rationale to be understandable to your company’s stakeholders. You also want to explore the results to identify any issues or potential biases.\n\nWhat should you do?",
    "options": [
      "A. 1. Use TensorFlow to generate and visualize features and statistics.\n\n2.Analyze the results together with the standard model evaluation metrics.",
      "B. 1. Use TensorFlow Profiler to visualize the model execution.\n\n2.Analyze the relationship between incorrect predictions and execution bottlenecks.",
      "C. 1. Use Vertex Explainable AI to generate example-based explanations.\n\n2.Visualize the results of sample inputs from the entire dataset together with the standard model evaluation metrics.",
      "D. 1. Use Vertex Explainable AI to generate feature attributions. Aggregate feature attributions over the entire dataset.\n\n2.Analyze the aggregation result together with the standard model evaluation metrics."
    ],
    "answer": "D. 1. Use Vertex Explainable AI to generate feature attributions. Aggregate feature attributions over the entire dataset.\n\n2.Analyze the aggregation result together with the standard model evaluation metrics."
  },
  {
    "question": "You trained a model, packaged it with a custom Docker container for serving, and deployed it to Vertex AI Model Registry. When you submit a batch prediction job, it fails with this error: \"Error model server never became ready. Please validate that your model file or container configuration are valid.\" There are no additional errors in the logs.\n\nWhat should you do?",
    "options": [
      "A. Add a logging configuration to your application to emit logs to Cloud Logging.",
      "B. Change the HTTP port in your model’s configuration to the default value of 8080.",
      "C. Change the healthRoute value in your model’s configuration to /healthcheck.",
      "D. Pull the Docker image locally, and use the docker run command to launch it locally. Use the docker logs command to explore the error logs."
    ],
    "answer": "D. Pull the Docker image locally, and use the docker run command to launch it locally. Use the docker logs command to explore the error logs."
  },
  {
    "question": "You are tasked with a dataset that encompasses customer transactions, and your objective is to construct an ML model for forecasting customer purchase patterns. Your plan involves creating the model within BigQuery ML and subsequently exporting it to Cloud Storage for online prediction. Upon reviewing the data, you observe the presence of categorical features such as product category and payment method.\n\nYour priority is to deploy the model swiftly. What steps should you take to achieve this goal?",
    "options": [
      "A. Use the TRANSFORM clause with the ML.ONE_HOT_ENCODER function on the categorical features at model creation and select the categorical and non-categorical features.",
      "B. Use the ML.ONE_HOT_ENCODER function on the categorical features and select the encoded categorical features and non-categorical features as inputs to create your model.",
      "C. Use the CREATE MODEL statement and select the categorical and non-categorical features.",
      "D. Use the ML.MULTI_HOT_ENCODER function on the categorical features, and select the encoded categorical features and non-categorical features as inputs to create your model."
    ],
    "answer": "C. Use the CREATE MODEL statement and select the categorical and non-categorical features."
  },
  {
    "question": "You have recently employed XGBoost to train a Python-based model designed for online serving. Your model prediction service will be accessed by a backend service built in Golang, operating on a Google Kubernetes Engine (GKE) cluster. Your model necessitates both pre-processing and post-processing steps, which must be executed during serving. Your primary objectives are to minimize code alterations, reduce infrastructure maintenance, and expedite the deployment of your model into a production environment. What steps should you take to accomplish these goals?",
    "options": [
      "A. Use FastAPI to implement an HTTP server. Create a Docker image that runs your HTTP server, and deploy it on your organization’s GKE cluster.",
      "B. Use FastAPI to implement an HTTP server. Create a Docker image that runs your HTTP server, Upload the image to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.",
      "C. Use the Predictor interface to implement a custom prediction routine. Build the custom container, upload the container to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.",
      "D. Use the XGBoost prebuilt serving container when importing the trained model into Vertex AI. Deploy the model to a Vertex AI endpoint. Work with the backend engineers to implement the pre- and postprocessing steps in the Golang backend service."
    ],
    "answer": "C. Use the Predictor interface to implement a custom prediction routine. Build the custom container, upload the container to Vertex AI Model Registry and deploy it to a Vertex AI endpoint."
  },
  {
    "question": "You have received a training-serving skew alert from a Vertex AI Model Monitoring job that is active in a production environment. In response, you have retrained the model using more up-to-date training data and subsequently redeployed it to the Vertex AI endpoint. Despite these actions, you continue to receive the same alert.\n\nWhat step should you take to address this situation?",
    "options": [
      "A. Update the model monitoring job to use a lower sampling rate.",
      "B. Update the model monitoring job to use the more recent training data that was used to retrain the model.",
      "C. Temporarily disable the alert. Enable the alert again after a sufficient amount of new production traffic has passed through the Vertex AI endpoint.",
      "D. Temporarily disable the alert until the model can be retrained again on newer training data. Retrain the model again after a sufficient amount of new production traffic has passed through the Vertex AI endpoint."
    ],
    "answer": "B. Update the model monitoring job to use the more recent training data that was used to retrain the model."
  },
  {
    "question": "You are employed at a social media company and have a requirement to create a no-code image classification model for an iOS mobile application, specifically designed for identifying fashion accessories. Your labeled dataset is stored in Cloud Storage. In this context, you aim to configure a training workflow that not only minimizes cost but also provides predictions with the lowest possible latency.\n\nHow should you proceed?",
    "options": [
      "A. Train the model using Vertex AI AutoML, and register the model in Vertex AI Model Registry. Configure your mobile application to send batch requests during prediction.",
      "B. Train the model using Vertex AI AutoML Edge, and export it as a Core ML model. Configure your mobile application to use the .mlmodel file directly.",
      "C. Train the model using Vertex AI AutoML Edge, and export the model as a TFLite model. Configure your mobile application to use the .tflite file directly.",
      "D. Train the model using Vertex AI AutoML, and expose the model as a Vertex AI endpoint. Configure your mobile application to invoke the endpoint during prediction."
    ],
    "answer": "B. Train the model using Vertex AI AutoML Edge, and export it as a Core ML model. Configure your mobile application to use the .mlmodel file directly."
  },
  {
    "question": "You work as an analyst at a large banking firm. You are developing a robust, scalable ML pipeline to train several regression and classification models. Your primary focus for the pipeline is model interpretability, and you want to quickly put the pipeline into production.\n\nWhat should you do?",
    "options": [
      "A. Use the Tabular Workflow for Wide & Deep provided by Vertex AI Pipelines to jointly train wide linear models and deep neural networks.",
      "B. Use Google Kubernetes Engine to construct a custom training pipeline for XGBoost-based models.",
      "C. Use the Tabular Workflow for TabNet in Vertex AI Pipelines to train attention-based models.",
      "D. Use Cloud Composer to establish training pipelines for custom deep learning-based models."
    ],
    "answer": "C. Use the Tabular Workflow for TabNet in Vertex AI Pipelines to train attention-based models."
  },
  {
    "question": "You are in the process of establishing a workflow for training and deploying your custom model in production. It's essential to maintain lineage information for your model and predictions.\n\nWhat steps should you take to achieve this?",
    "options": [
      "A. 1. Create a managed dataset in Vertex AI.\n2.Employ a Vertex AI training pipeline to train your model.\n3.Generate batch predictions using Vertex AI.",
      "B. 1. Utilize a Vertex AI Pipelines custom training job component to train your model.\n2.Produce predictions by using a Vertex AI Pipelines model batch predict component.",
      "C. 1. Upload your dataset to BigQuery.\n2.Utilize a Vertex AI custom training job to train your model.\n3.Generate predictions using Vertex AI SDK custom prediction routines.",
      "D. 1. Utilize Vertex AI Experiments for model training.\n2.Register your model in Vertex AI Model Registry.\n3.Generate batch predictions using Vertex AI."
    ],
    "answer": "D. 1. Utilize Vertex AI Experiments for model training.\n2.Register your model in Vertex AI Model Registry.\n3.Generate batch predictions using Vertex AI."
  },
  {
    "question": "You are developing a predictive maintenance model to proactively detect part defects in bridges, and you plan to utilize high-definition bridge images as inputs for your model. To effectively explain the model's output to the relevant stakeholders and enable them to take appropriate action, which approach should you use when building the model?",
    "options": [
      "A. Use scikit-learn to construct a tree-based model and employ SHAP (SHapley Additive exPlanations) values to explain the model's output.",
      "B. Use scikit-learn to build a tree-based model and employ partial dependence plots (PDP) to explain the model's output.",
      "C. Use TensorFlow to create a deep learning-based model and apply the Integrated Gradients method to explain the model's output.",
      "D. Use TensorFlow to create a deep learning-based model and utilize the sampled Shapley method to explain the model's output."
    ],
    "answer": "C. Use TensorFlow to create a deep learning-based model and apply the Integrated Gradients method to explain the model's output."
  },
  {
    "question": "You recently used BigQuery ML to train an Vertex AI AutoML regression model. You shared the results with your team and received positive feedback. You need to deploy your model for online prediction as quickly as possible.\n\nWhat should you do?",
    "options": [
      "A. Retrain the model using BigQuery ML and specify Vertex AI as the model registry. Deploy the model from Vertex AI Model Registry to a Vertex AI endpoint.",
      "B. Retrain the model using Vertex AI. Deploy the model from Vertex AI Model Registry to a Vertex AI endpoint.",
      "C. Alter the model using BigQuery ML and specify Vertex AI as the model registry. Deploy the model from Vertex AI Model Registry to a Vertex AI endpoint.",
      "D. Export the model from BigQuery ML to Cloud Storage. Import the model into Vertex AI Model Registry. Deploy the model to a Vertex AI endpoint."
    ],
    "answer": "C. Alter the model using BigQuery ML and specify Vertex AI as the model registry. Deploy the model from Vertex AI Model Registry to a Vertex AI endpoint."
  },
  {
    "question": "You work for a telecommunications company, and your task is to build a model for predicting which customers may fail to pay their next phone bill. The goal is to offer assistance to at-risk customers, such as service discounts and bill deadline extensions. Your dataset in BigQuery includes various features like Customer_id, Age, Salary, Sex, Average bill value, Number of phone calls in the last month, and Average duration of phone calls. Your objective is to address potential bias issues while maintaining model accuracy.\n\nWhat should you do?",
    "options": [
      "A. First, check for meaningful correlations between the sensitive features and other features. Then, train a BigQuery ML boosted trees classification model while excluding both the sensitive features and any significantly correlated features.",
      "B. Train a BigQuery ML boosted trees classification model using all available features. Afterward, use the ML.GLOBAL_EXPLAIN method to compute global attribution values for each feature. If any of the sensitive features have importance values exceeding a specified threshold, exclude those features and retrain the model.",
      "C. Train a BigQuery ML boosted trees classification model with all the features. Next, employ the ML.EXPLAIN_PREDICT method to determine attribution values for each feature per customer in a test set. If, for any individual customer, the importance value of any feature surpasses a predefined threshold, rebuild the model without that feature.",
      "D. Establish a fairness metric based on accuracy across the sensitive features. Train a BigQuery ML boosted trees classification model with all features. Then, utilize the trained model to make predictions on a test set. Join this data back with the sensitive features and compute a fairness metric to assess whether it meets your specified requirements."
    ],
    "answer": "D. Establish a fairness metric based on accuracy across the sensitive features. Train a BigQuery ML boosted trees classification model with all features. Then, utilize the trained model to make predictions on a test set. Join this data back with the sensitive features and compute a fairness metric to assess whether it meets your specified requirements."
  },
  {
    "question": "You are building a custom image classification model and plan to use Vertex AI Pipelines to implement the end-to-end training. Your dataset consists of images that need to be preprocessed before they can be used to train the model. The preprocessing steps include resizing the images, converting them to grayscale, and extracting features. You have already implemented some Python functions for the preprocessing tasks.\n\nWhich components should you use in your pipeline?",
    "options": [
      "A. DataprocSparkBatchOp and CustomTrainingJobOp",
      "B. DataflowPythonJobOp, WaitGcpResourcesOp, and CustomTrainingJobOp",
      "C. dsl.ParallelFor, dsl.component, and CustomTrainingJobOp",
      "D. ImageDatasetImportDataOp, dsl.component, and Vertex AI AutoMLImageTrainingJobRunOp"
    ],
    "answer": "B. DataflowPythonJobOp, WaitGcpResourcesOp, and CustomTrainingJobOp"
  },
  {
    "question": "You have established an ML pipeline featuring various input parameters, and your objective is to explore the trade-offs among different combinations of these parameters. The parameters in question include:\n - The input dataset\n - The maximum tree depth for the boosted tree regressor\n - The learning rate for the optimizer\n\nYou need to assess the pipeline's performance for the various parameter combinations, evaluating them in terms of F1 score, training time, and model complexity. It is essential for your methodology to be reproducible, and you aim to track all runs of the pipeline on a consistent platform. What steps should you take to achieve this?",
    "options": [
      "A. 1. Use BigQueryML to create a boosted tree regressor, and use the hyperparameter tuning capability.\n\n2.Configure the hyperparameter syntax to select different input datasets: max tree depths, and optimizer learning rates. Choose the grid search option.",
      "B. 1. Create a Vertex AI pipeline with a custom model training job as part of the pipeline. Configure the pipeline’s parameters to include those you are investigating.\n\n2.In the custom training step, use the Bayesian optimization method with F1 score as the target to maximize.",
      "C. 1. Create a Vertex AI Workbench notebook for each of the different input datasets.\n\n2.In each notebook, run different local training jobs with different combinations of the max tree depth and optimizer learning rate parameters.\n\n3.After each notebook finishes, append the results to a BigQuery table.",
      "D. 1. Create an experiment in Vertex AI Experiments.\n\n2.Create a Vertex AI pipeline with a custom model training job as part of the pipeline. Configure the pipeline’s parameters to include those you are investigating.\n\n3.Submit multiple runs to the same experiment, using different values for the parameters."
    ],
    "answer": "D. 1. Create an experiment in Vertex AI Experiments.\n\n2.Create a Vertex AI pipeline with a custom model training job as part of the pipeline. Configure the pipeline’s parameters to include those you are investigating.\n\n3.Submit multiple runs to the same experiment, using different values for the parameters."
  },
  {
    "question": "You are employed by a bank that adheres to rigorous data governance standards. Recently, you integrated a custom model designed to identify fraudulent transactions. Your intention is to configure your training code to access internal data through an API endpoint hosted within your project's network. Your primary concerns are to ensure the utmost security in accessing this data and to minimize the potential risk of data exfiltration.\n\nWhat steps should you take to achieve these objectives?",
    "options": [
      "A. Enable VPC Service Controls for peerings, and add Vertex AI to a service perimeter.",
      "B. Create a Cloud Run endpoint as a proxy to the data. Use Identity and Access Management (IAM) authentication to secure access to the endpoint from the training job.",
      "C. Configure VPC Peering with Vertex AI, and specify the network of the training job.",
      "D. Download the data to a Cloud Storage bucket before calling the training job."
    ],
    "answer": "A. Enable VPC Service Controls for peerings, and add Vertex AI to a service perimeter."
  },
  {
    "question": "You have recently deployed a scikit-learn model to a Vertex AI endpoint and are now in the process of testing it with live production traffic. While monitoring the endpoint, you've noticed that the number of requests per hour is twice as high as initially expected throughout the day. Your goal is to ensure that the endpoint can efficiently scale to meet increased demand in the future, thus preventing users from experiencing high latency.\n\nWhat actions should you take to address this situation?",
    "options": [
      "A. Deploy two models to the same endpoint, and distribute requests among them evenly",
      "B. Configure an appropriate minReplicaCount value based on expected baseline traffic",
      "C. Set the target utilization percentage in the autoscailngMetricSpecs configuration to a higher value",
      "D. Change the model’s machine type to one that utilizes GPUs"
    ],
    "answer": "B. Configure an appropriate minReplicaCount value based on expected baseline traffic"
  },
  {
    "question": "You are analyzing customer data for a healthcare organization that is stored in Cloud Storage. The data contains personally identifiable information (PII). You need to perform data exploration and preprocessing while ensuring the security and privacy of sensitive fields.\n\nWhat should you do?",
    "options": [
      "A. Use the Cloud Data Loss Prevention (DLP) API to de-identify the PII before performing data exploration and preprocessing.",
      "B. Use customer-managed encryption keys (CMEK) to encrypt the PII data at rest, and decrypt the PII data during data exploration and preprocessing.",
      "C. Use a VM inside a VPC Service Controls security perimeter to perform data exploration and preprocessing.",
      "D. Use Google-managed encryption keys to encrypt the PII data at rest, and decrypt the PII data during data exploration and preprocessing."
    ],
    "answer": "A. Use the Cloud Data Loss Prevention (DLP) API to de-identify the PII before performing data exploration and preprocessing."
  },
  {
    "question": "You are developing an ML pipeline using Vertex AI Pipelines. You want your pipeline to upload a new version of the XGBoost model to Vertex AI Model Registry and deploy it to Vertex AI Endpoints for online inference. You want to use the simplest approach.\n\nWhat should you do?",
    "options": [
      "A. Chain the Vertex AI ModelUploadOp and ModelDeployOp components together.",
      "B. Use the Vertex AI ModelEvaluationOp component to evaluate the model.",
      "C. Use the Vertex AI SDK for Python within a custom component based on a python:3.10 image.",
      "D. Use the Vertex AI REST API within a custom component based on a vertex-ai/prediction/xgboost-cpu image."
    ],
    "answer": "A. Chain the Vertex AI ModelUploadOp and ModelDeployOp components together."
  },
  {
    "question": "You have built a custom model that performs several memory-intensive preprocessing tasks before it makes a prediction. You deployed the model to a Vertex AI endpoint, and validated that results were received in a reasonable amount of time. After routing user traffic to the endpoint, you discover that the endpoint does not autoscale as expected when receiving multiple requests.\n\nWhat should you do?",
    "options": [
      "A. Use a machine type with more memory.",
      "B. Decrease the number of workers per machine.",
      "C. Increase the CPU utilization target in the autoscaling configurations.",
      "D. Decrease the CPU utilization target in the autoscaling configurations."
    ],
    "answer": "D. Decrease the CPU utilization target in the autoscaling configurations."
  },
  {
    "question": "You are collaborating on a model prototype with your team. You need to create a Vertex AI Workbench environment for the members of your team and also limit access to other employees in your project.\n\nWhat should you do?",
    "options": [
      "A.\n\nCreate a new service account and grant it the Notebook Viewer role.\n\nGrant the Service Account User role to each team member on the service account.\n\nGrant the Vertex AI User role to each team member.\n\nProvision a Vertex AI Workbench user-managed notebook instance that uses the new service account.",
      "B.\n\nGrant the Vertex AI User role to the default Compute Engine service account.\n\nGrant the Service Account User role to each team member on the default Compute Engine service account.\n\nProvision a Vertex AI Workbench user-managed notebook instance that uses the default Compute Engine service account.",
      "C.\n\nCreate a new service account and grant it the Vertex AI User role.\n\nGrant the Service Account User role to each team member on the service account.\n\nGrant the Notebook Viewer role to each team member.\n\nProvision a Vertex AI Workbench user-managed notebook instance that uses the new service account.",
      "D.\n\nGrant the Vertex AI User role to the primary team member.\n\nGrant the Notebook Viewer role to the other team members.\n\nProvision a Vertex AI Workbench user-managed notebook instance that uses the primary user’s account."
    ],
    "answer": "A.\n\nCreate a new service account and grant it the Notebook Viewer role.\n\nGrant the Service Account User role to each team member on the service account.\n\nGrant the Vertex AI User role to each team member.\n\nProvision a Vertex AI Workbench user-managed notebook instance that uses the new service account."
  }
]
questions_t5 = [
    {
    "question": "You are part of a food product company, and your historical sales data is stored in BigQuery. Your task is to utilize Vertex AI's custom training service to train multiple TensorFlow models, leveraging the data from BigQuery to predict future sales. In preparation for model experimentation, you plan to implement a data preprocessing algorithm that involves min-max scaling and bucketing for a significant number of features. Your aim is to keep preprocessing time, costs, and development efforts to a minimum.\n\nHow should you configure this workflow?",
    "options": [
      "A. Write the transformations into Spark that uses the spark-bigquery-connector, and use Dataproc to preprocess the data.",
      "B. Write SQL queries to transform the data in-place in BigQuery.",
      "C. Add the transformations as a preprocessing layer in the TensorFlow models.",
      "D. Create a Dataflow pipeline that uses the BigQuery lO connector to ingest the data, process it, and write it back to BigQuery."
    ],
    "answer": "B. Write SQL queries to transform the data in-place in BigQuery."
  },
  {
    "question": "You are an ML engineer at a retail company. You have built a model that predicts which coupon to offer an ecommerce customer at checkout based on the items in their cart. When a customer goes to checkout, your serving pipeline, which is hosted on Google Cloud, joins the customer's existing cart with a row in a BigQuery table that contains the customers' historic purchase behavior and uses that as the model's input. The web team is reporting that your model is returning predictions too slowly to load the coupon offer with the rest of the web page.\n\nHow should you speed up your model's predictions?",
    "options": [
      "A. Create a materialized view in BigQuery with the necessary data for predictions.",
      "B. Use a low latency database for the customers’ historic purchase behavior.",
      "C. Deploy your model to more instances behind a load balancer to distribute traffic.",
      "D. Attach an NVIDIA P100 GPU to your deployed model’s instance."
    ],
    "answer": "B. Use a low latency database for the customers’ historic purchase behavior."
  },
  {
    "question": "You work for a pet food company that manages an online forum. Customers upload photos of their pets on the forum to share with others. About 20 photos are uploaded daily. You want to automatically and in near real-time detect whether each uploaded photo has an animal. You want to prioritize time and minimize the cost of your application development and deployment.\n\nWhat should you do?",
    "options": [
      "A. Send user-submitted images to the Cloud Vision API. Use object localization to identify all objects in the image and compare the results against a list of animals.",
      "B. Download an object detection model from TensorFlow Hub. Deploy the model to a Vertex AI endpoint. Send new user-submitted images to the model endpoint to classify whether each photo has an animal.",
      "C. Manually label previously submitted images with bounding boxes around any animals. Build an Vertex AI AutoML object detection model by using Vertex AI. Deploy the model to a Vertex AI endpoint. Send new user-submitted images to your model endpoint to detect whether each photo has an animal.",
      "D. Manually label previously submitted images as having animals or not. Create an image dataset on Vertex AI. Train a classification model by using Vertex AutoML to distinguish the two classes. Deploy the model to a Vertex AI endpoint. Send new user-submitted images to your model endpoint to classify whether each photo has an animal."
    ],
    "answer": "A. Send user-submitted images to the Cloud Vision API. Use object localization to identify all objects in the image and compare the results against a list of animals."
  },
  {
    "question": "You are developing a training pipeline for a new XGBoost classification model based on tabular data. The data is stored in a BigQuery table. You need to complete the following steps:\n - Randomly split the data into training and evaluation datasets in a 65/35 ratio\n - Conduct feature engineering\n - Obtain metrics for the evaluation dataset\n - Compare models trained in different pipeline executions\n\nHow should you execute these steps?",
    "options": [
      "A. 1. Using Vertex AI Pipelines, add a component to divide the data into training and evaluation sets, and add another component for feature engineering.\n\n 2.Enable autologging of metrics in the training component.\n\n 3.Compare pipeline runs in Vertex AI Experiments.",
      "B. 1. Using Vertex AI Pipelines, add a component to divide the data into training and evaluation sets, and add another component for feature engineering.\n\n 2.Enable autologging of metrics in the training component.\n\n 3.Compare models using the artifacts’ lineage in Vertex ML Metadata.",
      "C. 1. In BigQuery ML, use the CREATE MODEL statement with BOOSTED_TREE_CLASSIFIER as the model type and use BigQuery to handle the data splits.\n\n 2.Use a SQL view to apply feature engineering and train the model using the data in that view.\n\n 3.Compare the evaluation metrics of the models by using a SQL query with the ML.TRAINING_INFO statement.",
      "D. 1. In BigQuery ML, use the CREATE MODEL statement with BOOSTED_TREE_CLASSIFIER as the model type and use BigQuery to handle the data splits.\n\n 2.Use ML TRANSFORM to specify the feature engineering transformations and train the model using the data in the table.\n\n 3.Compare the evaluation metrics of the models by using a SQL query with the ML.TRAINING_INFO statement."
    ],
    "answer": "A. 1. Using Vertex AI Pipelines, add a component to divide the data into training and evaluation sets, and add another component for feature engineering.\n\n 2.Enable autologging of metrics in the training component.\n\n 3.Compare pipeline runs in Vertex AI Experiments."
  },
  {
    "question": "You are employed by a magazine distribution company, and your task is to develop a predictive model for identifying customers who will renew their subscriptions for the upcoming year. You have utilized your company's historical data as the training dataset and have built a TensorFlow model, deploying it on Vertex AI. Now, your objective is to identify the most influential customer attribute for each prediction generated by the model.\n\nHow should you proceed?",
    "options": [
      "A. Stream prediction results to BigQuery. Use BigQuery’s CORR(X1, X2) function to calculate the Pearson correlation coefficient between each feature and the target variable.",
      "B. Use Vertex Explainable AI. Submit each prediction request with the explain keyword to retrieve feature attributions using the sampled Shapley method.",
      "C. Use Vertex AI Workbench user-managed notebooks to perform a Lasso regression analysis on your model, which will eliminate features that do not provide a strong signal.",
      "D. Use the What-If tool in Google Cloud to determine how your model will perform when individual features are excluded. Rank the feature importance in order of those that caused the most significant performance drop when removed from the model."
    ],
    "answer": "B. Use Vertex Explainable AI. Submit each prediction request with the explain keyword to retrieve feature attributions using the sampled Shapley method."
  },
  {
    "question": "You've been assigned the task of deploying prototype code into a production environment. The feature engineering component is written in PySpark and operates on Dataproc Serverless, while model training is conducted using a Vertex AI custom training job. These two steps are currently disjointed, requiring manual execution of model training after the feature engineering phase concludes. Your objective is to establish a scalable and maintainable production workflow that seamlessly connects and tracks these steps.\n\nWhat should you do?",
    "options": [
      "A. Create a Vertex AI Workbench notebook, utilize it to submit the Dataproc Serverless feature engineering job, and then submit the custom model training job within the same notebook. Execute the notebook cells sequentially to link the steps end-to-end.",
      "B. Create a Vertex AI Workbench notebook, initiate an Apache Spark context within the notebook, and execute the PySpark feature engineering code. Additionally, utilize the same notebook to execute the custom model training job in TensorFlow. Run the notebook cells sequentially to interconnect the steps from start to finish.",
      "C. Utilize the Kubeflow pipelines SDK to compose code specifying two components: the first being a Dataproc Serverless component responsible for initiating the feature engineering job, and the second being a custom component wrapped using the create_custom_training_job_from_component utility to launch the custom model training job. Create a Vertex AI Pipelines job to link and execute both of these components.",
      "D. Employ the Kubeflow pipelines SDK to draft code outlining two components: the first component initiates an Apache Spark context to execute the PySpark feature engineering code, and the second component runs the TensorFlow custom model training code. Establish a Vertex AI Pipelines job to interconnect and execute both of these components."
    ],
    "answer": "C. Utilize the Kubeflow pipelines SDK to compose code specifying two components: the first being a Dataproc Serverless component responsible for initiating the feature engineering job, and the second being a custom component wrapped using the create_custom_training_job_from_component utility to launch the custom model training job. Create a Vertex AI Pipelines job to link and execute both of these components."
  },
  {
    "question": "You work for an auto insurance company. You are preparing a proof-of-concept ML application that uses images of damaged vehicles to infer damaged parts. Your team has assembled a set of annotated images from damage claim documents in the company’s database. The annotations associated with each image consist of a bounding box for each identified damaged part and the part name. You have been given a sufficient budget to train models on Google Cloud. You need to quickly create an initial model.\n\nWhat should you do?",
    "options": [
      "A. Download a pre-trained object detection model from TensorFlow Hub. Fine-tune the model in Vertex AI Workbench by using the annotated image data.",
      "B. Train an object detection model in Vertex AI AutoML by using the annotated image data.",
      "C. Create a pipeline in Vertex AI Pipelines and configure the AutoMLTrainingJobRunOp component to train a custom object detection model by using the annotated image data.",
      "D. Train an object detection model in Vertex AI custom training by using the annotated image data."
    ],
    "answer": "B. Train an object detection model in Vertex AI AutoML by using the annotated image data."
  },
  {
    "question": "You are in the process of building a model aimed at identifying fraudulent credit card transactions, with a primary focus on enhancing detection capabilities since overlooking even a single fraudulent transaction could have serious consequences for the credit card holder. To train this model, you have employed Vertex AI AutoML, utilizing users' profile details and credit card transaction data.\n\nHowever, after the initial model training, you've observed that the model is falling short in detecting a significant number of fraudulent transactions.\n\nWhat modifications should you make to the training parameters in Vertex AI AutoML to enhance the model's performance? (Select two options.)",
    "options": [
      "A. Increase the score threshold",
      "B. Decrease the score threshold.",
      "C. Add more positive examples to the training set",
      "D. Add more negative examples to the training set",
      "E. Reduce the maximum number of node hours for training"
    ],
    "answer": [
      "B. Decrease the score threshold.",
      "C. Add more positive examples to the training set"
    ]
  },
  {
    "question": "You are employed at a retail company and have developed a Vertex AI forecast model that produces monthly item sales predictions. Now, you aim to swiftly generate a report that explains how the model calculates these predictions. You possess one month of recent actual sales data that was not part of the training dataset.\n\nWhat steps should you take to generate data for your report?",
    "options": [
      "A. Create a batch prediction job using the actual sales data and compare the predictions to the actuals in the report.",
      "B. Create a batch prediction job using the actual sales data and configure the job settings to generate feature attributions. Then, compare the results in the report.",
      "C. Generate counterfactual examples using the actual sales data. Subsequently, create a batch prediction job by using both the actual sales data and the counterfactual examples, and compare the results in the report.",
      "D. Train another model using the same training dataset as the original, excluding some columns. Then, utilize the actual sales data to create one batch prediction job with the new model and another with the original model. Finally, compare the two sets of predictions in the report."
    ],
    "answer": "B. Create a batch prediction job using the actual sales data and configure the job settings to generate feature attributions. Then, compare the results in the report."
  },
  {
    "question": "You need to develop a custom TensorFlow model for online predictions with training data stored in BigQuery. You want to apply instance-level data transformations to the data consistently during both model training and serving.\n\nHow should you configure the preprocessing routine?",
    "options": [
      "A. Create a BigQuery script to preprocess the data and save the result to another BigQuery table.",
      "B. Create a Vertex AI Pipelines pipeline to read the data from BigQuery and perform preprocessing using a custom preprocessing component.",
      "C. Develop a preprocessing function that reads and transforms the data from BigQuery. Create a Vertex AI custom prediction routine that calls the preprocessing function during serving.",
      "D. Implement an Apache Beam pipeline that reads data from BigQuery and preprocesses it using TensorFlow Transform and Dataflow."
    ],
    "answer": "D. Implement an Apache Beam pipeline that reads data from BigQuery and preprocesses it using TensorFlow Transform and Dataflow."
  },
  {
    "question": "You are creating a model training pipeline to predict sentiment scores from text-based product reviews. You want to have control over how the model parameters are tuned, and you will deploy the model to an endpoint after it has been trained. You will use Vertex AI Pipelines to run the pipeline. You need to decide which Google Cloud pipeline components to use.\n\nWhat components should you choose?",
    "options": [
      "A. TabularDatasetCreateOp, CustomTrainingJobOp, and EndpointCreateOp",
      "B. TextDatasetCreateOp, AutoMLTextTrainingOp, and EndpointCreateOp",
      "C. TabularDatasetCreateOp, AutoMLTextTrainingOp, and ModelDeployOp",
      "D. TextDatasetCreateOp, CustomTrainingJobOp, and ModelDeployOp"
    ],
    "answer": "D. TextDatasetCreateOp, CustomTrainingJobOp, and ModelDeployOp"
  },
  {
    "question": "Your company manages an ecommerce website. You developed an ML model that recommends additional products to users in near real time based on items currently in the user’s cart. The workflow will include the following processes:\n\n - The website will send a Pub/Sub message with the relevant data and then receive a message with the prediction from Pub/Sub\n - Predictions will be stored in BigQuery\n - The model will be stored in a Cloud Storage bucket and will be updated frequently\n\nYou want to minimize prediction latency and the effort required to update the model. How should you reconfigure the architecture?",
    "options": [
      "A. Write a Cloud Function that loads the model into memory for prediction. Configure the function to be triggered when messages are sent to Pub/Sub.",
      "B. Create a pipeline in Vertex AI Pipelines that performs preprocessing, prediction, and postprocessing. Configure the pipeline to be triggered by a Cloud Function when messages are sent to Pub/Sub.",
      "C. Expose the model as a Vertex AI endpoint. Write a custom DoFn in a Dataflow job that calls the endpoint for prediction.",
      "D. Use the RunInference API with WatchFilePattern in a Dataflow job that wraps around the model and serves predictions."
    ],
    "answer": "D. Use the RunInference API with WatchFilePattern in a Dataflow job that wraps around the model and serves predictions."
  },
  {
    "question": "You have recently trained a scikit-learn model that you plan to deploy on Vertex AI. This model will support both online and batch prediction. You need to preprocess input data for model inference. You want to package the model for deployment while minimizing additional code.",
    "options": [
        "A. 1. Upload your model to the Vertex AI Model Registry by using a prebuilt scikit-learn prediction container.\n\n2.Deploy your model to Vertex AI Endpoints, and create a Vertex AI batch prediction job that uses the instanceConfig.instanceType setting to transform your input data.",
        "B. 1. Wrap your model in a custom prediction routine (CPR), and build a container image from the CPR local model.\n\n2.Upload your scikit-learn model container to Vertex AI Model Registry.\n\n3.Deploy your model to Vertex AI Endpoints, and create a Vertex AI batch prediction job.",
        "C. 1. Create a custom container for your scikit-learn model.\n\n2.Define a custom serving function for your model.\n\n3.Upload your model and custom container to Vertex AI Model Registry.\n\n4.Deploy your model to Vertex AI Endpoints, and create a Vertex AI batch prediction job.",
        "D. 1. Create a custom container for your scikit-learn model.\n\n2.Upload your model and custom container to Vertex AI Model Registry.\n\n3.Deploy your model to Vertex AI Endpoints, and create a Vertex AI batch prediction job that uses the instanceConfig.instanceType setting to transform your input data."
    ],
    "answer": "B. 1. Wrap your model in a custom prediction routine (CPR), and build a container image from the CPR local model.\n\n2.Upload your scikit-learn model container to Vertex AI Model Registry.\n\n3.Deploy your model to Vertex AI Endpoints, and create a Vertex AI batch prediction job."
  },
  {
    "question": "You are in the process of implementing a batch inference ML pipeline within Google Cloud. The model, developed using TensorFlow, is stored in SavedModel format within Cloud Storage. Your task involves applying this model to a historical dataset, which comprises a substantial 10 TB of data stored within a BigQuery table.\n\nHow should you proceed to perform the inference effectively?",
    "options": [
      "A. Export the historical data to Cloud Storage in Avro format. Configure a Vertex AI batch prediction job to generate predictions for the exported data",
      "B. Import the TensorFlow model by using the CREATE MODEL statement in BigQuery ML. Apply the historical data to the TensorFlow model",
      "C. Export the historical data to Cloud Storage in CSV format. Configure a Vertex AI batch prediction job to generate predictions for the exported data",
      "D. Configure a Vertex AI batch prediction job to apply the model to the historical data in BigQuery"
    ],
    "answer": "D. Configure a Vertex AI batch prediction job to apply the model to the historical data in BigQuery"
  },
  {
    "question": "You work at a bank, and your task is to develop a credit risk model to support loan application decisions. You've chosen to implement this model using a neural network in TensorFlow. Regulatory requirements mandate that you should be able to explain the model's predictions based on its features. Additionally, when the model is deployed, you want to continuously monitor its performance over time. To achieve this, you have opted to utilize Vertex AI for both model development and deployment.\n\nWhat should be your course of action?",
    "options": [
      "A. Utilize Vertex Explainable AI with the sampled Shapley method, and enable Vertex AI Model Monitoring to check for feature distribution drift.",
      "B. Utilize Vertex Explainable AI with the sampled Shapley method, and enable Vertex AI Model Monitoring to check for feature distribution skew.",
      "C. Utilize Vertex Explainable AI with the XRAI method, and enable Vertex AI Model Monitoring to check for feature distribution drift.",
      "D. Utilize Vertex Explainable AI with the XRAI method, and enable Vertex AI Model Monitoring to check for feature distribution skew."
    ],
    "answer": "A. Utilize Vertex Explainable AI with the sampled Shapley method, and enable Vertex AI Model Monitoring to check for feature distribution drift."
  },
  {
    "question": "You are using Keras and TensorFlow to develop a fraud detection model. Records of customer transactions are stored in a large table in BigQuery. You need to preprocess these records in a cost-effective and efficient way before you use them to train the model. The trained model will be used to perform batch inference in BigQuery.\n\nHow should you implement the preprocessing workflow?",
    "options": [
      "A. Implement a preprocessing pipeline by using Apache Spark, and run the pipeline on Dataproc. Save the preprocessed data as CSV files in a Cloud Storage bucket.",
      "B. Load the data into a pandas DataFrame. Implement the preprocessing steps using pandas transformations, and train the model directly on the DataFrame.",
      "C. Perform preprocessing in BigQuery by using SQL. Use the BigQueryClient in TensorFlow to read the data directly from BigQuery.",
      "D. Implement a preprocessing pipeline by using Apache Beam, and run the pipeline on Dataflow. Save the preprocessed data as CSV files in a Cloud Storage bucket."
    ],
    "answer": "C. Perform preprocessing in BigQuery by using SQL. Use the BigQueryClient in TensorFlow to read the data directly from BigQuery."
  },
  {
    "question": "You work for a retail company that is using a regression model built with BigQuery ML to predict product sales. This model is being used to serve online predictions. Recently you developed a new version of the model that uses a different architecture (custom model). Initial analysis revealed that both models are performing as expected. You want to deploy the new version of the model to production and monitor the performance over the next two months. You need to minimize the impact to the existing and future model users.\n\nHow should you deploy the model?",
    "options": [
      "A. Import the new model to the same Vertex AI Model Registry as a different version of the existing model. Deploy the new model to the same Vertex AI endpoint as the existing model, and use traffic splitting to route 95% of production traffic to the BigQuery ML model and 5% of production traffic to the new model.",
      "B. Import the new model to the same Vertex AI Model Registry as the existing model. Deploy the models to one Vertex AI endpoint. Route 95% of production traffic to the BigQuery ML model and 5% of production traffic to the new model.",
      "C. Import the new model to the same Vertex AI Model Registry as the existing model. Deploy each model to a separate Vertex AI endpoint.",
      "D. Deploy the new model to a separate Vertex AI endpoint. Create a Cloud Run service that routes the prediction requests to the corresponding endpoints based on the input feature values."
    ],
    "answer": "A. Import the new model to the same Vertex AI Model Registry as a different version of the existing model. Deploy the new model to the same Vertex AI endpoint as the existing model, and use traffic splitting to route 95% of production traffic to the BigQuery ML model and 5% of production traffic to the new model."
  },
  {
    "question": "You work for a large retailer, and you need to build a model to predict customer churn. The company has a dataset of historical customer data, including customer demographics purchase history, and website activity. You need to create the model in BigQuery ML and thoroughly evaluate its performance.\n\nWhat should you do?",
    "options": [
      "A. Create a linear regression model in BigQuery ML, and register the model in Vertex AI Model Registry. Evaluate the model performance in Vertex AI.",
      "B. Create a logistic regression model in BigQuery ML and register the model in Vertex AI Model Registry. Evaluate the model performance in Vertex AI.",
      "C. Create a linear regression model in BigQuery ML. Use the ML.EVALUATE function to evaluate the model performance.",
      "D. Create a logistic regression model in BigQuery ML. Use the ML.CONFUSION_MATRIX function to evaluate the model performance."
    ],
    "answer": "B. Create a logistic regression model in BigQuery ML and register the model in Vertex AI Model Registry. Evaluate the model performance in Vertex AI."
  },
  {
    "question": "You've created a custom model in Vertex AI to predict user churn rate for your application. Vertex AI Model Monitoring is used for skew detection, and your training data in BigQuery includes two types of features: demographic and behavioral. Recently, you found that two separate models, each trained on one of these feature sets, outperform the original model. Now, you want to set up a new model monitoring pipeline that directs traffic to both models while maintaining consistent prediction-sampling rates and monitoring frequencies. You also aim to minimize management overhead.\n\nWhat should be your approach?",
    "options": [
      "A. Leave the training dataset unchanged. Deploy the models to two separate endpoints and initiate two Vertex AI Model Monitoring jobs with appropriate feature-threshold parameters.",
      "B. Keep the training dataset intact. Deploy both models to the same endpoint and launch a Vertex AI Model Monitoring job with a monitoring configuration from a file that accounts for model IDs and feature selections.",
      "C. Divide the training dataset into two tables based on demographic and behavioral features. Deploy the models to two separate endpoints and initiate two Vertex AI Model Monitoring jobs.",
      "D. Split the training dataset into two tables based on demographic and behavioral features. Deploy both models to the same endpoint and submit a Vertex AI Model Monitoring job with a monitoring configuration from a file that considers model IDs and training datasets."
    ],
    "answer": "B. Keep the training dataset intact. Deploy both models to the same endpoint and launch a Vertex AI Model Monitoring job with a monitoring configuration from a file that accounts for model IDs and feature selections."
  },
  {
    "question": "As an ML engineer at a manufacturing company, you're currently working on a predictive maintenance project. The goal is to create a classification model that predicts whether a critical machine will experience a failure within the next three days. This predictive capability allows the repair team to address potential issues before they lead to a breakdown. While routine maintenance for the machine is cost-effective, a failure can result in significant expenses.\n\nYou've trained multiple binary classifiers to make predictions about the machine's failure, where a prediction of 1 indicates the model foresees a failure.\n\nNow, during the evaluation phase on a separate dataset, you face the decision of selecting a model that emphasizes detection. However, you also need to ensure that over 50% of the maintenance tasks initiated by your model are genuinely related to impending machine failures.\n\nWhich model should you opt for to achieve this balance?",
    "options": [
      "A. The model with the highest area under the receiver operating characteristic curve (AUC ROC) and precision greater than 0.5",
      "B. The model with the lowest root mean squared error (RMSE) and recall greater than 0.5.",
      "C. The model with the highest recall where precision is greater than 0.5.",
      "D. The model with the highest precision where recall is greater than 0.5."
    ],
    "answer": "C. The model with the highest recall where precision is greater than 0.5."
  },
  {
    "question": "You are in the process of deploying a new version of a model to a production Vertex AI endpoint that is actively serving user traffic. Your goal is to direct all user traffic to the new model while minimizing any disruption to your application.\n\nHow should you proceed to achieve this objective?",
    "options": [
      "A.\n\nCreate a new endpoint\n\nCreate a new model. Set it as the default version. Upload the model to Vertex AI Model Registry\n\nDeploy the new model to the new endpoint\n\nUpdate Cloud DNS to point to the new endpoint",
      "B.\n\nCreate a new endpoint\nCreate a new model. Set the parentModel parameter to the model ID of the currently deployed model and set it as the default version.\n\nUpload the model to Vertex AI Model Registry\n\nDeploy the new model to the new endpoint, and set the new model to 100% of the traffic.",
      "C.\n\nCreate a new model. Set the parentModel parameter to the model ID of the currently deployed model. Upload the model to Vertex AI Model Registry.\n\nDeploy the new model to the existing endpoint, and set the new model to 100% of the traffic",
      "D.\n\nCreate a new model. Set it as the default version. Upload the model to Vertex AI Model Registry\n\nDeploy the new model to the existing endpoint"
    ],
    "answer": "C.\n\nCreate a new model. Set the parentModel parameter to the model ID of the currently deployed model. Upload the model to Vertex AI Model Registry.\n\nDeploy the new model to the existing endpoint, and set the new model to 100% of the traffic"
  },
  {
    "question": "You work for a company that captures live video footage of checkout areas in their retail stores. Your task is to build a model to detect the number of customers waiting for service in near real-time. You aim to create this solution quickly and with minimal effort.\n\nWhat approach should you take to build the model?",
    "options": [
      "A. Utilize the Vertex AI Vision Occupancy Analytics model.",
      "B. Utilize the Vertex AI Vision Person/vehicle detector model.",
      "C. Train an Vertex AI AutoML object detection model on an annotated dataset using Vertex AutoML.",
      "D. Train a Seq2Seq+ object detection model on an annotated dataset using Vertex AutoML."
    ],
    "answer": "A. Utilize the Vertex AI Vision Occupancy Analytics model."
  },
  {
    "question": "You recently developed a wide and deep model in TensorFlow, and you generated training datasets using a SQL script in BigQuery for preprocessing raw data. You now need to create a training pipeline for weekly model retraining, which will generate daily recommendations. Your goal is to minimize model development and training time.\n\nHow should you develop the training pipeline?",
    "options": [
      "A. Use the Kubeflow Pipelines SDK to implement the pipeline. Employ the BigQueryJobOp component to run the preprocessing script and the CustomTrainingJobOp component to launch a Vertex AI training job.",
      "B. Use the Kubeflow Pipelines SDK to implement the pipeline. Utilize the DataflowPythonJobOp component for data preprocessing and the CustomTrainingJobOp component to initiate a Vertex AI training job.",
      "C. Use the TensorFlow Extended SDK to implement the pipeline. Use the ExampleGen component with the BigQuery executor for data ingestion, the Transform component for data preprocessing, and the Trainer component to launch a Vertex AI training job.",
      "D. Use the TensorFlow Extended SDK to implement the pipeline. Integrate the preprocessing steps into the input_fn of the model. Use the ExampleGen component with the BigQuery executor for data ingestion and the Trainer component to initiate a Vertex AI training job."
    ],
    "answer": "A. Use the Kubeflow Pipelines SDK to implement the pipeline. Employ the BigQueryJobOp component to run the preprocessing script and the CustomTrainingJobOp component to launch a Vertex AI training job."
  },
  {
    "question": "You have trained a model using data that was preprocessed in a batch Dataflow pipeline, and now you need real-time inference while ensuring consistent data preprocessing between training and serving.\n\nWhat should you do?",
    "options": [
      "A. Perform data validation to ensure that the input data to the pipeline matches the input data format for the endpoint.",
      "B. Refactor the transformation code in the batch data pipeline so that it can be used outside of the pipeline and employ the same code in the endpoint.",
      "C. Refactor the transformation code in the batch data pipeline so that it can be used outside of the pipeline, and share this code with the endpoint's end users.",
      "D. Batch the real-time requests using a time window, preprocess the batched requests using the Dataflow pipeline, and then send the preprocessed requests to the endpoint."
    ],
    "answer": "B. Refactor the transformation code in the batch data pipeline so that it can be used outside of the pipeline and employ the same code in the endpoint."
  },
  {
    "question": "You are employed at a pharmaceutical company in Canada, and your team has developed a BigQuery ML model for predicting the monthly flu infection count in Canada. Weather data is updated weekly, while flu infection statistics are updated monthly. Your task is to establish a model retraining policy that minimizes expenses.\n\nWhat would you recommend?",
    "options": [
      "A. Download the weather and flu data on a weekly basis. Configure Cloud Scheduler to trigger a Vertex AI pipeline for weekly model retraining.",
      "B. Download the weather and flu data on a monthly basis. Configure Cloud Scheduler to trigger a Vertex AI pipeline for monthly model retraining.",
      "C. Download the weather and flu data weekly. Configure Cloud Scheduler to trigger a Vertex AI pipeline for model retraining every month.",
      "D. Download weather data weekly and flu data monthly. Deploy the model on a Vertex AI endpoint with feature drift monitoring and retrain the model if a monitoring alert is triggered."
    ],
    "answer": "D. Download weather data weekly and flu data monthly. Deploy the model on a Vertex AI endpoint with feature drift monitoring and retrain the model if a monitoring alert is triggered."
  },
  {
    "question": "You have created a Vertex AI pipeline that automates custom model training. You want to add a pipeline component that enables your team to collaborate most easily when running different executions and comparing metrics both visually and programmatically.\n\nWhat should you do?",
    "options": [
      "A. Add a component to the Vertex AI pipeline that logs metrics to a BigQuery table. Query the table to compare different executions of the pipeline. Connect BigQuery to Looker Studio to visualize metrics.",
      "B. Add a component to the Vertex AI pipeline that logs metrics to a BigQuery table. Load the table into a pandas DataFrame to compare different executions of the pipeline. Use Matplotlib to visualize metrics.",
      "C. Add a component to the Vertex AI pipeline that logs metrics to Vertex ML Metadata. Use Vertex AI Experiments to compare different executions of the pipeline. Use Vertex AI TensorBoard to visualize metrics.",
      "D. Add a component to the Vertex AI pipeline that logs metrics to Vertex ML Metadata. Load the Vertex ML Metadata into a pandas DataFrame to compare different executions of the pipeline. Use Matplotlib to visualize metrics."
    ],
    "answer": "C. Add a component to the Vertex AI pipeline that logs metrics to Vertex ML Metadata. Use Vertex AI Experiments to compare different executions of the pipeline. Use Vertex AI TensorBoard to visualize metrics."
  },
  {
    "question": "You work for a delivery company. You need to design a system that stores and manages features such as parcels delivered and truck locations over time. The system must retrieve the features with low latency and feed those features into a model for online prediction. The data science team will retrieve historical data at a specific point in time for model training. You want to store the features with minimal effort.\n\nWhat should you do?",
    "options": [
      "A. Store features in Bigtable as key/value data.",
      "B. Store features in Vertex AI Feature Store.",
      "C. Store features as a Vertex AI dataset, and use those features to train the models hosted in Vertex AI endpoints.",
      "D. Store features in BigQuery timestamp-partitioned tables, and use the BigQuery Storage Read API to serve the features."
    ],
    "answer": "B. Store features in Vertex AI Feature Store."
  },
  {
    "question": "You have constructed a Vertex AI pipeline consisting of two key stages. The initial step involves the preprocessing of a substantial 10 TB dataset, completing this task within approximately 1 hour, and then saving the resulting data in a Cloud Storage bucket. The subsequent step utilizes this preprocessed data to train a model. Your current objective is to make adjustments to the model's code, facilitating the testing of different algorithms. Throughout this process, you aim to reduce both the pipeline's execution time and cost while keeping any alterations to the pipeline itself to a minimum.\n\nWhat actions should you take to meet these goals?",
    "options": [
      "A. Add a pipeline parameter and an additional pipeline step. Depending on the parameter value, the pipeline step conducts or skips data preprocessing, and starts model training.",
      "B. Create another pipeline without the preprocessing step, and hardcode the preprocessed Cloud Storage file location for model training.",
      "C. Configure a machine with more CPU and RAM from the compute-optimized machine family for the data preprocessing step.",
      "D. Enable caching for the pipeline job, and disable caching for the model training step."
    ],
    "answer": "D. Enable caching for the pipeline job, and disable caching for the model training step."
  },
  {
    "question": "You are in the process of developing an MLOps platform to automate your company's machine learning experiments and model retraining. You require an efficient way to manage the artifacts for multiple pipelines.\n\nHow should you go about organizing the pipelines' artifacts?",
    "options": [
      "A. Store the parameters in Cloud SQL, while keeping the models' source code and binaries in GitHub.",
      "B. Store the parameters in Cloud SQL, the models' source code in GitHub, and the models' binaries in Cloud Storage.",
      "C. Store the parameters in Vertex ML Metadata, alongside the models' source code in GitHub, and the models' binaries in Cloud Storage.",
      "D. Store the parameters in Vertex ML Metadata, while placing the models' source code and binaries in GitHub."
    ],
    "answer": "C. Store the parameters in Vertex ML Metadata, alongside the models' source code in GitHub, and the models' binaries in Cloud Storage."
  },
  {
    "question": "You work on a team that builds state-of-the-art deep learning models by using the TensorFlow framework. Your team runs multiple ML experiments each week, which makes it difficult to track the experiment runs. You want a simple approach to effectively track, visualize, and debug ML experiment runs on Google Cloud while minimizing any overhead code.\n\nHow should you proceed?",
    "options": [
      "A. Set up Vertex AI Experiments to track metrics and parameters. Configure Vertex AI TensorBoard for visualization.",
      "B. Set up a Cloud Function to write and save metrics files to a Cloud Storage bucket. Configure a Google Cloud VM to host TensorBoard locally for visualization.",
      "C. Set up a Vertex AI Workbench notebook instance. Use the instance to save metrics data in a Cloud Storage bucket and to host TensorBoard locally for visualization.",
      "D. Set up a Cloud Function to write and save metrics files to a BigQuery table. Configure a Google Cloud VM to host TensorBoard locally for visualization."
    ],
    "answer": "A. Set up Vertex AI Experiments to track metrics and parameters. Configure Vertex AI TensorBoard for visualization."
  },
  {
    "question": "You have recently trained an XGBoost model that you intend to deploy for online inference in production. Before sending a predict request to your model's binary, you need to perform a straightforward data preprocessing step. This step should expose a REST API that can accept requests within your internal VPC Service Controls and return predictions. Your goal is to configure this preprocessing step while minimizing both cost and effort.\n\nWhat should you do?",
    "options": [
      "A. Store a pickled model in Cloud Storage. Develop a Flask-based application, package the application in a custom container image, and then deploy the model to Vertex AI Endpoints.",
      "B. Create a Flask-based application, package the application and a pickled model in a custom container image, and deploy the model to Vertex AI Endpoints.",
      "C. Develop a custom predictor class based on XGBoost Predictor from the Vertex AI SDK, package it along with a pickled model in a custom container image based on a Vertex built-in image, and deploy the model to Vertex AI Endpoints.",
      "D. Design a custom predictor class based on XGBoost Predictor from the Vertex AI SDK, package the handler in a custom container image based on a Vertex built-in container image. Store a pickled model in Cloud Storage and deploy the model to Vertex AI Endpoints."
    ],
    "answer": "D. Design a custom predictor class based on XGBoost Predictor from the Vertex AI SDK, package the handler in a custom container image based on a Vertex built-in container image. Store a pickled model in Cloud Storage and deploy the model to Vertex AI Endpoints."
  },
  {
    "question": "You've developed a custom ML model using scikit-learn, and you've encountered longer-than-expected training times. To address this issue, you've chosen to migrate your model to Vertex AI Training and are now looking for initial steps to enhance the training efficiency. What should be your first approach?",
    "options": [
      "A. Train your model in a distributed mode using multiple Compute Engine VMs.",
      "B. Train your model using Vertex AI Training with CPUs.",
      "C. Migrate your model to TensorFlow, and train it using Vertex AI Training.",
      "D. Train your model using Vertex AI Training with GPUs."
    ],
    "answer": "B. Train your model using Vertex AI Training with CPUs."
  },
  {
    "question": "You work for an online grocery store. You recently developed a custom ML model that recommends a recipe when a user arrives at the website. You chose the machine type on the Vertex AI endpoint to optimize costs by using the queries per second (QPS) that the model can serve, and you deployed it on a single machine with 8 vCPUs and no accelerators.\n\nA holiday season is approaching and you anticipate four times more traffic during this time than the typical daily traffic. You need to ensure that the model can scale efficiently to the increased demand.\n\nWhat should you do?",
    "options": [
      "A.\n\n1. Maintain the same machine type on the endpoint.\n2. Set up a monitoring job and an alert for CPU usage.\n3. If you receive an alert, add a compute node to the endpoint.",
      "B.\n\n1. Change the machine type on the endpoint to have 32 vCPUs.\n2. Set up a monitoring job and an alert for CPU usage.\n3. If you receive an alert, scale the vCPUs further as needed.",
      "C.\n\n1. Maintain the same machine type on the endpoint Configure the endpoint to enable autoscaling based on vCPU usage.\n2. Set up a monitoring job and an alert for CPU usage.\n3. If you receive an alert, investigate the cause.",
      "D.\n\n1. Change the machine type on the endpoint to have a GPU. Configure the endpoint to enable autoscaling based on the GPU usage.\n2. Set up a monitoring job and an alert for GPU usage.\n3. If you receive an alert, investigate the cause."
    ],
    "answer": "C.\n\n1. Maintain the same machine type on the endpoint Configure the endpoint to enable autoscaling based on vCPU usage.\n2. Set up a monitoring job and an alert for CPU usage.\n3. If you receive an alert, investigate the cause."
  },
  {
    "question": "You are using Vertex AI and TensorFlow to develop a custom image classification model. You need the model’s decisions and the rationale to be understandable to your company’s stakeholders. You also want to explore the results to identify any issues or potential biases.\n\nWhat should you do?",
    "options": [
        "A. 1. Use TensorFlow to generate and visualize features and statistics.\n\n2.Analyze the results together with the standard model evaluation metrics.",
        "B. 1. Use TensorFlow Profiler to visualize the model execution.\n\n2.Analyze the relationship between incorrect predictions and execution bottlenecks.",
        "C. 1. Use Vertex Explainable AI to generate example-based explanations.\n\n2.Visualize the results of sample inputs from the entire dataset together with the standard model evaluation metrics.",
        "D. 1. Use Vertex Explainable AI to generate feature attributions. Aggregate feature attributions over the entire dataset.\n\n2.Analyze the aggregation result together with the standard model evaluation metrics."
    ],
    "answer": "D. 1. Use Vertex Explainable AI to generate feature attributions. Aggregate feature attributions over the entire dataset.\n\n2.Analyze the aggregation result together with the standard model evaluation metrics."
  },
  {
    "question": "You are required to construct an image classification model utilizing an extensive dataset that is stored within a Cloud Storage bucket. How should you proceed with this task?",
    "options": [
      "A. Use Vertex AI Pipelines with the Kubeflow Pipelines SDK to create a pipeline that reads the images from Cloud Storage and trains the model.",
      "B. Use Vertex AI Pipelines with TensorFlow Extended (TFX) to create a pipeline that reads the images from Cloud Storage and trains the model.",
      "C. Import the labeled images as a managed dataset in Vertex AI and use AutoML to train the model.",
      "D. Convert the image dataset to a tabular format using Dataflow Load the data into BigQuery and use BigQuery ML to train the model."
    ],
    "answer": "C. Import the labeled images as a managed dataset in Vertex AI and use AutoML to train the model."
  },
  {
    "question": "You are employed at a bank and have a custom tabular ML model provided by the bank's vendor. Unfortunately, the training data for this model is sensitive and unavailable. The model is packaged as a Vertex AI Model serving container, and it accepts a string as input for each prediction instance. Within these strings, feature values are separated by commas. Your objective is to deploy this model into production for online predictions while also monitoring the feature distribution over time with minimal effort.\n\nWhat steps should you take to achieve this?",
    "options": [
        "A. 1. Upload the model to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.\n\n2.Create a Vertex AI Model Monitoring job with feature drift detection as the monitoring objective, and provide an instance schema.",
        "B. 1. Upload the model to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.\n\n2.Create a Vertex AI Model Monitoring job with feature skew detection as the monitoring objective, and provide an instance schema.",
        "C. 1. Refactor the serving container to accept key-value pairs as an input format.\n\n2.Upload the model to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.\n\n3.Create a Vertex AI Model Monitoring job with feature drift detection as the monitoring objective.",
        "D. 1. Refactor the serving container to accept key-value pairs as an input format.\n\n2.Upload the model to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.\n\n3.Create a Vertex AI Model Monitoring job with feature skew detection as the monitoring objective."
    ],
    "answer": "A. 1. Upload the model to Vertex AI Model Registry and deploy it to a Vertex AI endpoint.\n\n2.Create a Vertex AI Model Monitoring job with feature drift detection as the monitoring objective, and provide an instance schema."
  },
  {
    "question": "You are training a custom language model for your company using a large dataset, and you plan to use the Reduction Server strategy on Vertex AI. You need to configure the worker pools for the distributed training job.\n\nWhat should you do?",
    "options": [
      "A. Configure the machines of the first two worker pools to have GPUs and use a container image where your training code runs. Configure the third worker pool to have GPUs and use the reductionserver container image.",
      "B. Configure the machines of the first two worker pools to have GPUs and use a container image where your training code runs. Configure the third worker pool without accelerators, use the reductionserver container image, and choose a machine type that prioritizes bandwidth.",
      "C. Configure the machines of the first two worker pools to have TPUs and use a container image where your training code runs. Configure the third worker pool without accelerators, use the reductionserver container image, and choose a machine type that prioritizes bandwidth.",
      "D. Configure the machines of the first two worker pools to have TPUs and use a container image where your training code runs. Configure the third worker pool to have TPUs and use the reductionserver container image."
    ],
    "answer": "B. Configure the machines of the first two worker pools to have GPUs and use a container image where your training code runs. Configure the third worker pool without accelerators, use the reductionserver container image, and choose a machine type that prioritizes bandwidth."
  },
  {
    "question": "You work at a mobile gaming startup that creates online multiplayer games. Recently, your company observed an increase in players cheating in the games, leading to a loss of revenue and a poor user experience. You build a binary classification model to determine whether a player cheated after a completed game session and then send a message to other downstream systems to ban the player that cheated. Your model has performed well during testing, and you now need to deploy the model to production. You want your serving solution to provide immediate classifications after a completed game session to avoid further loss of revenue.\n\nWhat should you do?",
    "options": [
      "A. Import the model into Vertex AI Model Registry. Use the Vertex Batch Prediction service to run batch inference jobs.",
      "B. Save the model files in a Cloud Storage bucket. Create a Cloud Function to read the model files and make online inference requests on the Cloud Function.",
      "C. Save the model files in a VM. Load the model files each time there is a prediction request, and run an inference job on the VM.",
      "D. Import the model into Vertex AI Model Registry. Create a Vertex AI endpoint that hosts the model and make online inference requests."
    ],
    "answer": "D. Import the model into Vertex AI Model Registry. Create a Vertex AI endpoint that hosts the model and make online inference requests."
  },
  {
    "question": "Your team has deployed a model to a Vertex AI endpoint, and you've established a Vertex AI pipeline that streamlines the model training process, triggered by a Cloud Function. Your primary goals are to keep the model up-to-date while also minimizing retraining costs.\n\nHow should you configure the retraining process?",
    "options": [
      "A. Configure Pub/Sub to notify the Cloud Function when a sufficient amount of new data becomes available.",
      "B. Configure a Cloud Scheduler job to trigger the Cloud Function at a predetermined frequency that aligns with your team's budget.",
      "C. Enable model monitoring on the Vertex AI endpoint and configure Pub/Sub to notify the Cloud Function when anomalies are detected.",
      "D. Enable model monitoring on the Vertex AI endpoint and configure Pub/Sub to notify the Cloud Function when feature drift is detected."
    ],
    "answer": "D. Enable model monitoring on the Vertex AI endpoint and configure Pub/Sub to notify the Cloud Function when feature drift is detected."
  },
  {
    "question": "You are employed at a prominent healthcare company, tasked with creating advanced algorithms for a range of applications. Your dataset consists of unstructured text data with specialized annotations. Your objective is to extract and categorize different medical expressions with these annotations.\n\nWhat course of action should you take?",
    "options": [
      "A. Utilize the Healthcare Natural Language API for medical entity extraction.",
      "B. Employ a BERT-based model to fine-tune a medical entity extraction model.",
      "C. Utilize Vertex AI AutoML Entity Extraction to train a medical entity extraction model.",
      "D. Develop a customized medical entity extraction model using TensorFlow."
    ],
    "answer": "C. Utilize Vertex AI AutoML Entity Extraction to train a medical entity extraction model."
  },
  {
    "question": "You are investigating the root cause of a misclassification error made by one of your models. You used Vertex AI Pipelines to train and deploy the model. The pipeline reads data from BigQuery. creates a copy of the data in Cloud Storage in TFRecord format, trains the model in Vertex AI Training on that copy, and deploys the model to a Vertex AI endpoint. You have identified the specific version of that model that misclassified, and you need to recover the data this model was trained on.\n\nHow should you find that copy of the data?",
    "options": [
      "A. Use Vertex AI Feature Store. Modify the pipeline to use the feature store, and ensure that all training data is stored in it. Search the feature store for the data used for the training.",
      "B. Use the lineage feature of Vertex AI Metadata to find the model artifact. Determine the version of the model and identify the step that creates the data copy and search in the metadata for its location.",
      "C. Use the logging features in the Vertex AI endpoint to determine the timestamp of the model’s deployment. Find the pipeline run at that timestamp. Identify the step that creates the data copy, and search in the logs for its location.",
      "D. Find the job ID in Vertex AI Training corresponding to the training for the model. Search in the logs of that job for the data used for the training."
    ],
    "answer": "B. Use the lineage feature of Vertex AI Metadata to find the model artifact. Determine the version of the model and identify the step that creates the data copy and search in the metadata for its location."
  },
  {
    "question": "You work for a company that sells corporate electronic products to thousands of businesses worldwide. Your company stores historical customer data in BigQuery. You need to build a model that predicts customer lifetime value over the next three years. You want to use the simplest approach to build the model and you want to have access to visualization tools.\n\nWhat should you do?",
    "options": [
      "A. Create a Vertex AI Workbench notebook to perform exploratory data analysis. Use IPython magics to create a new BigQuery table with input features. Use the BigQuery console to run the CREATE MODEL statement. Validate the results by using the ML.EVALUATE and ML.PREDICT statements.",
      "B. Run the CREATE MODEL statement from the BigQuery console to create an Vertex AI AutoML model. Validate the results by using the ML.EVALUATE and ML.PREDICT statements.",
      "C. Create a Vertex AI Workbench notebook to perform exploratory data analysis and create input features. Save the features as a CSV file in Cloud Storage. Import the CSV file as a new BigQuery table. Use the BigQuery console to run the CREATE MODEL statement. Validate the results by using the ML.EVALUATE and ML.PREDICT statements.",
      "D. Create a Vertex AI Workbench notebook to perform exploratory data analysis. Use IPython magics to create a new BigQuery table with input features, create the model, and validate the results by using the CREATE MODEL, ML.EVALUATE, and ML.PREDICT statements."
    ],
    "answer": "D. Create a Vertex AI Workbench notebook to perform exploratory data analysis. Use IPython magics to create a new BigQuery table with input features, create the model, and validate the results by using the CREATE MODEL, ML.EVALUATE, and ML.PREDICT statements."
  },
  {
    "question": "You are eager to train an AutoML model for predicting house prices using a compact public dataset stored in BigQuery. Your primary objective is to prepare the data efficiently, opting for the simplest and most straightforward approach.\n\nWhat step should you take to achieve this goal?",
    "options": [
      "A. Write a query that preprocesses the data by using BigQuery and creates a new table. Create a Vertex AI managed dataset with the new table as the data source.",
      "B. Use Dataflow to preprocess the data. Write the output in TFRecord format to a Cloud Storage bucket.",
      "C. Write a query that preprocesses the data by using BigQuery. Export the query results as CSV files, and use those files to create a Vertex AI managed dataset.",
      "D. Use a Vertex AI Workbench notebook instance to preprocess the data by using the pandas library. Export the data as CSV files, and use those files to create a Vertex AI managed dataset."
    ],
    "answer": "A. Write a query that preprocesses the data by using BigQuery and creates a new table. Create a Vertex AI managed dataset with the new table as the data source."
  },
  {
    "question": "You've recently implemented a pipeline within Vertex AI Pipelines, which is responsible for training and deploying a model to a Vertex AI endpoint to serve real-time traffic. Your objective is to maintain an ongoing process of experimentation and iteration to enhance model performance. To facilitate this, you intend to employ Cloud Build for continuous integration and continuous deployment (CI/CD). Your ultimate goal is to efficiently and swiftly deploy new pipelines into production while minimizing the risk of potential disruptions to the existing production environment due to new pipeline implementations.\n\nWhat step should you take to achieve this?",
    "options": [
      "A. Set up a CI/CD pipeline that builds and tests your source code, if the tests are successful, use the Google Cloud console to upload the built container to Artifact Registry and upload the compiled pipeline to Vertex AI Pipelines.",
      "B. Set up a CI/CD pipeline that builds your source code and then deploys built artifacts into a pre-production environment. Run unit tests in the pre-production environment, if the tests are successfully deployed the pipeline to production.",
      "C. Set up a CI/CD pipeline that builds and tests your source code and then deploys built artifacts into a pre-production environment. After a successful pipeline run in the pre-production environment, deploy the pipeline to production.",
      "D. Set up a CI/CD pipeline that builds and tests your source code and then deploys built artifacts into a pre-production environment. After a successful pipeline run in the pre-production environment, rebuild the source code and deploy the artifacts to production."
    ],
    "answer": "C. Set up a CI/CD pipeline that builds and tests your source code and then deploys built artifacts into a pre-production environment. After a successful pipeline run in the pre-production environment, deploy the pipeline to production."
  },
  {
    "question": "You are in the midst of training an ML model on a sizable dataset, and you are utilizing a TPU (Tensor Processing Unit) to accelerate the training process. However, you've noticed that the training is proceeding slower than expected, and upon investigation, you've determined that the TPU is not fully utilizing its capacity.\n\nWhat actions should you take to address this issue?",
    "options": [
      "A. Increase the learning rate",
      "B. Increase the number of epochs",
      "C. Decrease the learning rate",
      "D. Increase the batch size"
    ],
    "answer": "D. Increase the batch size"
  },
  {
    "question": "You work for a retail company, and your task is to develop a model for predicting whether a customer will make a purchase on a given day. Your team has processed the company's sales data and created a table with specific columns, including customer ID, product ID, date, days since the last purchase, average purchase frequency, and a binary class indicating whether the customer made a purchase on the date in question. Your objective is to interpret the results of your model for individual predictions.\n\nWhat is the recommended approach?",
    "options": [
      "A. Create a BigQuery table. Use BigQuery ML to construct a boosted tree classifier. Examine the partition rules within the trees to comprehend how each prediction is guided through the tree structure.",
      "B. Create a Vertex AI tabular dataset. Train an AutoML model for predicting customer purchases. Deploy the model to a Vertex AI endpoint and enable feature attributions. Utilize the \"explain\" method to acquire feature attribution values for each individual prediction.",
      "C. Create a BigQuery table. Employ BigQuery ML to develop a logistic regression classification model. Interpret the feature importance by examining the coefficients of the model, with higher coefficient values indicating greater importance.",
      "D. Create a Vertex AI tabular dataset. Train an AutoML model for predicting customer purchases. Deploy the model to a Vertex AI endpoint. For each prediction, activate L1 regularization to identify non-informative features."
    ],
    "answer": "B. Create a Vertex AI tabular dataset. Train an AutoML model for predicting customer purchases. Deploy the model to a Vertex AI endpoint and enable feature attributions. Utilize the \"explain\" method to acquire feature attribution values for each individual prediction."
  },
  {
    "question": "You work for a hospital aiming to optimize its operation scheduling process. To predict the number of beds needed for patients based on scheduled surgeries, you have one year of data organized in 365 rows, including variables like the number of scheduled surgeries, the number of beds occupied, and the date for each day. Your goal is to maximize the speed of model development and testing.\n\nWhat should you do?",
    "options": [
      "A. Create a BigQuery table. Use BigQuery ML to build a regression model with the number of beds as the target variable and features like the number of scheduled surgeries and date-related features (e.g., day of the week) as predictors.",
      "B. Create a BigQuery table. Use BigQuery ML to build an ARIMA model with the number of beds as the target variable and date as the time variable.",
      "C. Create a Vertex AI tabular dataset. Train an AutoML regression model with the number of beds as the target variable and features like the number of scheduled minor surgeries and date-related features (e.g., day of the week) as predictors.",
      "D. Create a Vertex AI tabular dataset. Train a Vertex AI AutoML Forecasting model with the number of beds as the target variable, the number of scheduled surgeries as a covariate, and date as the time variable."
    ],
    "answer": "D. Create a Vertex AI tabular dataset. Train a Vertex AI AutoML Forecasting model with the number of beds as the target variable, the number of scheduled surgeries as a covariate, and date as the time variable."
  },
  {
    "question": "You work for an online retailer. Your company has a few thousand short lifecycle products. Your company has five years of sales data stored in BigQuery. You have been asked to build a model that will make monthly sales predictions for each product. You want to use a solution that can be implemented quickly with minimal effort.\n\nWhat should you do?",
    "options": [
      "A. Use Prophet on Vertex AI Training to build a custom model.",
      "B. Use BigQuery ML to build a statistical ARIMA_PLUS model.",
      "C. Use Vertex AI Forecast to build a NN-based model.",
      "D. Use TensorFlow on Vertex AI Training to build a custom model."
    ],
    "answer": "B. Use BigQuery ML to build a statistical ARIMA_PLUS model."
  },
  {
    "question": "You are training models in Vertex AI using data that spans across multiple Google Cloud projects. You need to find, track, and compare the performance of the different versions of your models.\n\nWhich Google Cloud services should you include in your ML workflow?",
    "options": [
      "A. Dataplex, Vertex AI Feature Store, and Vertex AI TensorBoard",
      "B. Vertex AI Pipelines, Vertex AI Feature Store, and Vertex AI Experiments",
      "C. Dataplex, Vertex AI Experiments, and Vertex AI ML Metadata",
      "D. Vertex AI Pipelines, Vertex AI Experiments, and Vertex AI Metadata"
    ],
    "answer": "D. Vertex AI Pipelines, Vertex AI Experiments, and Vertex AI Metadata"
  },
  {
    "question": "You work at an ecommerce startup. You need to create a customer churn prediction model. Your company’s recent sales records are stored in a BigQuery table. You want to understand how your initial model is making predictions. You also want to iterate on the model as quickly as possible while minimizing cost.\n\nHow should you build your first model?",
    "options": [
      "A. Export the data to a Cloud Storage bucket. Load the data into a pandas DataFrame on Vertex AI Workbench and train a logistic regression model with scikit-learn.",
      "B. Create a tf.data.Dataset by using the TensorFlow BigQueryClient. Implement a deep neural network in TensorFlow.",
      "C. Prepare the data in BigQuery and associate the data with a Vertex AI dataset. Create an AutoMLTabularTrainingJob to tram a classification model.",
      "D. Export the data to a Cloud Storage bucket. Create a tf.data.Dataset to read the data from Cloud Storage. Implement a deep neural network in TensorFlow."
    ],
    "answer": "C. Prepare the data in BigQuery and associate the data with a Vertex AI dataset. Create an AutoMLTabularTrainingJob to tram a classification model."
  }
]
questions_t6 = [
    {
    "question": "You need to train an XGBoost model on a small dataset and your training code has custom dependencies. To minimize the startup time of your training job, how should you configure your Vertex AI custom training job?",
    "options": [
      "A. Store the data in a Cloud Storage bucket and create a custom container with your training application, which reads the data from Cloud Storage to train the model.",
      "B. Use the XGBoost prebuilt custom container, create a Python source distribution that includes the data and installs dependencies at runtime, and train the model by loading the data into a pandas DataFrame.",
      "C. Create a custom container that includes the data and train the model by loading the data into a pandas DataFrame.",
      "D. Store the data in a Cloud Storage bucket, use the XGBoost prebuilt custom container for your training application, create a Python source distribution that installs the dependencies at runtime, and read the data from Cloud Storage to train the model."
    ],
    "answer": "A. Store the data in a Cloud Storage bucket and create a custom container with your training application, which reads the data from Cloud Storage to train the model."
  },
  {
    "question": "You work at a gaming startup and have several terabytes of structured data stored in Cloud Storage, including gameplay time, user metadata, and game metadata. You need to build a model to recommend new games to users with minimal coding. What should you do?",
    "options": [
      "A. Load the data into BigQuery and use BigQuery ML to train an Autoencoder model.",
      "B. Load the data into BigQuery and use BigQuery ML to train a matrix factorization model.",
      "C. Load the data into a Vertex AI Workbench notebook and use TensorFlow to train a two-tower model.",
      "D. Load the data into a Vertex AI Workbench notebook and use TensorFlow to train a matrix factorization model."
    ],
    "answer": "B. Load the data into BigQuery and use BigQuery ML to train a matrix factorization model."
  },
  {
    "question": "You're working on an ML model within a Vertex AI Workbench notebook and aim to track artifacts and compare models during experimentation while efficiently transitioning successful experiments to production as you iterate on your model implementation.\n\nWhat should you do?",
    "options": [
      "A.\n\n1. Initialize the Vertex SDK with the experiment name. Log parameters and metrics for each experiment, and attach dataset and model artifacts as inputs and outputs to each execution.\n\n2. After a successful experiment, create a Vertex AI pipeline.",
      "B.\n\n1. Initialize the Vertex SDK with the experiment name. Log parameters and metrics for each experiment, save your dataset to a Cloud Storage bucket, and upload the models to Vertex AI Model Registry.\n\n2. After a successful experiment, create a Vertex AI pipeline.",
      "C.\n\n1. Create a Vertex AI pipeline with parameters tracked as arguments to your PipelineJob. Utilize Metrics, Model, and Dataset artifact types from the Kubeflow Pipelines DSL as inputs and outputs of the pipeline components.\n\n2. Associate the pipeline with your experiment upon job submission.",
      "D.\n\n1. Create a Vertex AI pipeline. Use Dataset and Model artifact types from the Kubeflow Pipelines DSL as inputs and outputs of the pipeline components.\n\n2. Within your training component, employ the Vertex AI SDK to create an experiment run. Configure the log_params and log_metrics functions to track parameters and metrics of your experiment."
    ],
    "answer": "A.\n\n1. Initialize the Vertex SDK with the experiment name. Log parameters and metrics for each experiment, and attach dataset and model artifacts as inputs and outputs to each execution.\n\n2. After a successful experiment, create a Vertex AI pipeline."
  },
  {
    "question": "You're working on an application to assist users with meal planning, aiming to employ machine learning for extracting ingredients and kitchen cookware from a corpus of recipes saved as unstructured text files.\n\nWhat approach should you take?",
    "options": [
      "A. Set up a text dataset on Vertex AI tailored for entity extraction. Establish two entities named \"ingredient\" and \"cookware\" and label a minimum of 200 instances for each. Train an AutoML entity extraction model to identify occurrences of these entities. Assess model performance using a holdout dataset.",
      "B. Establish a multi-label text classification dataset on Vertex AI. Create a testing dataset and label each recipe with its corresponding ingredients and cookware. Train a multi-class classification model and evaluate its performance using a holdout dataset.",
      "C. Utilize the Entity Analysis feature of the Natural Language API to extract ingredients and cookware from each recipe. Evaluate the model's performance using a prelabeled dataset.",
      "D. Establish a text dataset on Vertex AI optimized for entity extraction. Generate entities corresponding to the different ingredients and cookware present. Train an AutoML entity extraction model to recognize these entities and evaluate its performance using a holdout dataset."
    ],
    "answer": "A. Set up a text dataset on Vertex AI tailored for entity extraction. Establish two entities named \"ingredient\" and \"cookware\" and label a minimum of 200 instances for each. Train an AutoML entity extraction model to identify occurrences of these entities. Assess model performance using a holdout dataset."
  },
  {
    "question": "You've developed a model utilizing BigQuery ML for linear regression and aim to retrain it weekly with the cumulative data while minimizing both development effort and scheduling costs.\n\nWhat approach should you take?",
    "options": [
      "A. Utilize BigQuery's scheduling service to periodically execute the model retraining query.",
      "B. Construct a pipeline within Vertex AI Pipelines to execute the retraining query and employ the Cloud Scheduler API to schedule its weekly execution.",
      "C. Employ Cloud Scheduler to trigger a Cloud Function weekly, which in turn runs the query for model retraining.",
      "D. Use the BigQuery API Connector alongside Cloud Scheduler to trigger Workflows weekly, facilitating the model retraining process."
    ],
    "answer": "A. Utilize BigQuery's scheduling service to periodically execute the model retraining query."
  },
  {
    "question": "You're developing a model to enhance your company's online advertising campaigns, aiming to create a dataset for model training while avoiding the creation or reinforcement of unfair bias.\n\nWhat steps should you take? (Choose two.)",
    "options": [
      "A. Include a comprehensive set of demographic features.",
      "B. Include only the demographic groups that most frequently interact with advertisements.",
      "C. Collect a random sample of production traffic to build the training dataset.",
      "D. Collect a stratified sample of production traffic to build the training dataset.",
      "E. Conduct fairness tests across sensitive categories and demographics on the trained model."
    ],
    "answer": [
      "D. Collect a stratified sample of production traffic to build the training dataset.",
      "E. Conduct fairness tests across sensitive categories and demographics on the trained model."
    ]
  },
  {
    "question": "You're training and deploying updated versions of a regression model with tabular data using Vertex AI Pipelines, Vertex AI Training, Vertex AI Experiments, and Vertex AI Endpoints. The deployed model resides in a Vertex AI endpoint, and users access it via this endpoint. You aim to receive an email notification when significant changes occur in the feature data distribution, prompting you to retrigger the training pipeline and deploy an updated model. What should you do?",
    "options": [
      "A. Utilize Vertex AI Model Monitoring. Enable prediction drift monitoring on the endpoint, and specify a notification email.",
      "B. Create a logs-based alert in Cloud Logging using the logs from the Vertex AI endpoint. Configure Cloud Logging to send an email when the alert triggers.",
      "C. Set up a logs-based metric in Cloud Monitoring and define a threshold alert for the metric. Configure Cloud Monitoring to send an email notification upon alert activation.",
      "D. Export the container logs of the endpoint to BigQuery. Develop a Cloud Function to execute a SQL query over the exported logs and send an email notification. Use Cloud Scheduler to trigger the Cloud Function."
    ],
    "answer": "A. Utilize Vertex AI Model Monitoring. Enable prediction drift monitoring on the endpoint, and specify a notification email."
  },
  {
    "question": "You've recently deployed a model to a Vertex AI endpoint and configured online serving in Vertex AI Feature Store. As part of your setup, you've scheduled a daily batch ingestion job to update your feature store. However, during these batch ingestion processes, you notice high CPU utilization in your feature store's online serving nodes, leading to increased feature retrieval latency. To enhance online serving performance during these daily batch ingestion tasks, what should you do?",
    "options": [
      "A. Schedule an increase in the number of online serving nodes in your feature store prior to the batch ingestion jobs.",
      "B. Enable autoscaling for the online serving nodes in your feature store.",
      "C. Activate autoscaling for the prediction nodes of your DeployedModel in the Vertex AI endpoint.",
      "D. Adjust the worker_count parameter in the ImportFeatureValues request of your batch ingestion job."
    ],
    "answer": "A. Schedule an increase in the number of online serving nodes in your feature store prior to the batch ingestion jobs."
  },
  {
    "question": "You are developing an ML model to identify your company’s products in images. You have access to over one million images in a Cloud Storage bucket. You plan to experiment with different TensorFlow models by using Vertex AI Training. You need to read images at scale during training while minimizing data I/O bottlenecks.\n\nWhat should you do?",
    "options": [
      "A. Load the images directly into the Vertex AI compute nodes by using Cloud Storage FUSE. Read the images by using the tf.data.Dataset.from_tensor_slices function.",
      "B. Create a Vertex AI managed dataset from your image data. Access the AIP_TRAINING_DATA_URI environment variable to read the images by using the tf.data.Dataset.list_files function.",
      "C. Convert the images to TFRecords and store them in a Cloud Storage bucket. Read the TFRecords by using the tf.data.TFRecordDataset function.",
      "D. Store the URLs of the images in a CSV file. Read the file by using the tf.data.experimental.CsvDataset function."
    ],
    "answer": "C. Convert the images to TFRecords and store them in a Cloud Storage bucket. Read the TFRecords by using the tf.data.TFRecordDataset function."
  },
  {
    "question": "You are developing an ML model that predicts the cost of used automobiles based on data such as location, condition, model type, color, and engine/battery efficiency. The data is updated every night. Car dealerships will use the model to determine appropriate car prices. You created a Vertex AI pipeline that reads the data, splits the data into training/evaluation/test sets, performs feature engineering, trains the model using the training dataset, and validates the model using the evaluation dataset. You need to configure a retraining workflow that minimizes cost.\n\nWhat should you do?",
    "options": [
      "A. Compare the training and evaluation losses of the current run. If the losses are similar, deploy the model to a Vertex AI endpoint. Configure a cron job to redeploy the pipeline every night.",
      "B. Compare the training and evaluation losses of the current run. If the losses are similar, deploy the model to a Vertex AI endpoint with training/serving skew threshold model monitoring. When the model monitoring threshold is triggered, redeploy the pipeline.",
      "C. Compare the results to the evaluation results from a previous run. If the performance improved, deploy the model to a Vertex AI endpoint. Configure a cron job to redeploy the pipeline every night.",
      "D. Compare the results to the evaluation results from a previous run. If the performance improved, deploy the model to a Vertex AI endpoint with training/serving skew threshold model monitoring. When the model monitoring threshold is triggered, redeploy the pipeline."
    ],
    "answer": "D. Compare the results to the evaluation results from a previous run. If the performance improved, deploy the model to a Vertex AI endpoint with training/serving skew threshold model monitoring. When the model monitoring threshold is triggered, redeploy the pipeline."
  },
  {
    "question": "You've recently set up a new Google Cloud project and successfully tested the submission of a Vertex AI Pipeline job from Cloud Shell. Now, you're attempting to run your code from a Vertex AI Workbench user-managed notebook instance, but the job fails with an insufficient permissions error. What action should you take?",
    "options": [
      "A. Ensure that the Workbench instance you created is in the same region as the Vertex AI Pipelines resources you intend to use.",
      "B. Confirm that the Vertex AI Workbench instance is on the same subnetwork as the Vertex AI Pipeline resources you intend to use.",
      "C. Verify that the Vertex AI Workbench instance is assigned the Identity and Access Management (IAM) Vertex AI User role.",
      "D. Verify that the Vertex AI Workbench instance is assigned the Identity and Access Management (IAM) Notebooks Runner role."
    ],
    "answer": "C. Verify that the Vertex AI Workbench instance is assigned the Identity and Access Management (IAM) Vertex AI User role."
  },
  {
    "question": "You are designing an ML pipeline for data processing, model training, and deployment using various Google Cloud services, and you have developed code for each task. Given the high frequency of new file arrivals, you need an orchestration layer that initiates only when new files appear in your Cloud Storage bucket. Additionally, you want to minimize compute node costs. What approach should you take?",
    "options": [
      "A. Create a pipeline in Vertex AI Pipelines and configure its initial step to check for new files since the last pipeline execution, using the scheduler API to run the pipeline periodically.",
      "B. Create a Cloud Function triggered by Cloud Storage to activate a Cloud Composer directed acyclic graph (DAG).",
      "C. Create a pipeline in Vertex AI Pipelines and a Cloud Function triggered by Cloud Storage to deploy the pipeline.",
      "D. Deploy a Cloud Composer directed acyclic graph (DAG) with a GCSObjectUpdateSensor that triggers when a new file is added to the Cloud Storage bucket."
    ],
    "answer": "C. Create a pipeline in Vertex AI Pipelines and a Cloud Function triggered by Cloud Storage to deploy the pipeline."
  },
  {
    "question": "You intend to migrate a scikit-learn classifier model to TensorFlow, planning to train the TensorFlow model using the same training set as the scikit-learn model. Subsequently, you aim to compare their performances using a common test set, logging the evaluation metrics of each model manually using the Vertex AI Python SDK, and comparing them based on their F1 scores and confusion matrices.\n\nHow should you log the metrics?",
    "options": [
      "A. Utilize the aiplatform.log_classification_metrics function to log the F1 score, and employ the aiplatform.log_metrics function to log the confusion matrix.",
      "B. Utilize the aiplatform.log_classification_metrics function to log both the F1 score and the confusion matrix.",
      "C. Utilize the aiplatform.log_metrics function to log both the F1 score and the confusion matrix.",
      "D. Utilize the aiplatform.log_metrics function to log the F1 score, and employ the aiplatform.log_classification_metrics function to log the confusion matrix."
    ],
    "answer": "D. Utilize the aiplatform.log_metrics function to log the F1 score, and employ the aiplatform.log_classification_metrics function to log the confusion matrix."
  },
  {
    "question": "You recently trained an XGBoost model using tabular data and plan to make it available as an HTTP microservice for internal use. Anticipating a low volume of incoming requests, you seek the most efficient method to deploy the model with minimal effort and latency. What is your best option?",
    "options": [
      "A. Deploy the model to BigQuery ML using the CREATE MODEL statement with the BOOSTED_TREE_REGRESSOR option, and call the BigQuery API from the microservice.",
      "B. Develop a Flask-based application, package it in a custom container on Vertex AI, and deploy it to Vertex AI Endpoints.",
      "C. Create a Flask-based application, package it in a Docker image, and deploy it to Google Kubernetes Engine using Autopilot mode.",
      "D. Utilize a prebuilt XGBoost Vertex container to create a model, and deploy it to Vertex AI Endpoints."
    ],
    "answer": "D. Utilize a prebuilt XGBoost Vertex container to create a model, and deploy it to Vertex AI Endpoints."
  },
  {
    "question": "You've trained an XGBoost model for deployment on Vertex AI for online prediction. As you upload your model to Vertex AI Model Registry, you need to configure the explanation method for serving online prediction requests with minimal latency. Additionally, you want to receive alerts when feature attributions of the model significantly change over time.\n\nWhat steps should you take?",
    "options": [
      "A.\n\n1. Specify sampled Shapley as the explanation method with a path count of 5.\n\n2. Deploy the model to Vertex AI Endpoints.\n\nCreate a Model Monitoring job using prediction drift as the monitoring objective.",
      "B.\n\n1. Specify Integrated Gradients as the explanation method with a path count of 5.\n\n2. Deploy the model to Vertex AI Endpoints.\n\nCreate a Model Monitoring job using prediction drift as the monitoring objective.",
      "C.\n\n1. Specify sampled Shapley as the explanation method with a path count of 50.\n\n2. Deploy the model to Vertex AI Endpoints.\n\nCreate a Model Monitoring job using training-serving skew as the monitoring objective.",
      "D.\n\n1. Specify Integrated Gradients as the explanation method with a path count of 50.\n\n2. Deploy the model to Vertex AI Endpoints.\n\nCreate a Model Monitoring job using training-serving skew as the monitoring objective."
    ],
    "answer": "A.\n\n1. Specify sampled Shapley as the explanation method with a path count of 5.\n\n2. Deploy the model to Vertex AI Endpoints.\n\nCreate a Model Monitoring job using prediction drift as the monitoring objective."
  },
  {
    "question": "You're part of a rapidly growing social media company, where your team builds TensorFlow recommender models on an on-premises CPU cluster. With billions of historical user events and 100,000 categorical features in the data, you've observed increasing model training times as the data grows. Now, you're planning to migrate the models to Google Cloud and seek the most scalable approach that minimizes training time.\n\nWhat should you do?",
    "options": [
      "A. Deploy the training jobs using TPU VMs with TPUv3 Pod slices, and utilize the TPUEmbedding API.",
      "B. Deploy the training jobs within an autoscaling Google Kubernetes Engine cluster with CPUs.",
      "C. Deploy a matrix factorization model training job using BigQuery ML.",
      "D. Deploy the training jobs using Compute Engine instances with A100 GPUs, and utilize the tf.nn.embedding_lookup API."
    ],
    "answer": "A. Deploy the training jobs using TPU VMs with TPUv3 Pod slices, and utilize the TPUEmbedding API."
  },
  {
    "question": "You work for a large bank with an application hosted on Google Cloud, operating in the US and Singapore. You've developed a PyTorch model, a three-layer perceptron, to classify transactions as potentially fraudulent. This model uses both numerical and categorical features, with hashing conducted within the model itself.\n\nCurrently, the model is deployed in the us-central1 region on nl-highcpu-16 machines, delivering predictions with a median response latency of 40 ms. To reduce latency, particularly for users in Singapore who are experiencing longer delays, what should you do?",
    "options": [
      "A. Attach an NVIDIA T4 GPU to the machines currently used for online inference.",
      "B. Switch the machines used for online inference to nl-highcpu-32.",
      "C. Deploy the model to Vertex AI private endpoints in both the us-central1 and asia-southeast1 regions, allowing the application to select the most suitable endpoint.",
      "D. Establish another Vertex AI endpoint in the asia-southeast1 region, enabling the application to select the most appropriate endpoint."
    ],
    "answer": "D. Establish another Vertex AI endpoint in the asia-southeast1 region, enabling the application to select the most appropriate endpoint."
  },
  {
    "question": "You are building a recommendation engine for an online clothing store, with historical customer transaction data stored in BigQuery and Cloud Storage. To conduct exploratory data analysis (EDA), preprocessing, and model training iteratively while experimenting with different algorithms, you aim to minimize costs and development efforts.\n\nHow should you configure the environment?",
    "options": [
      "A. Set up a Vertex AI Workbench user-managed notebook using the default VM instance, and utilize the %%bigquery magic commands in Jupyter to query the tables.",
      "B. Establish a Vertex AI Workbench managed notebook for direct browsing and querying of tables through the JupyterLab interface.",
      "C. Create a Vertex AI Workbench user-managed notebook within a Dataproc Hub, employing the %%bigquery magic commands in Jupyter for table querying.",
      "D. Deploy a Vertex AI Workbench managed notebook on a Dataproc cluster, utilizing the spark-bigquery-connector to access the tables."
    ],
    "answer": "A. Set up a Vertex AI Workbench user-managed notebook using the default VM instance, and utilize the %%bigquery magic commands in Jupyter to query the tables."
  },
  {
    "question": "You're tasked with constructing a model to predict churn probability for customers at a retail company. It's crucial for the predictions to be interpretable, enabling the development of targeted marketing campaigns for at-risk customers. What approach should you take?",
    "options": [
      "A. Develop a random forest regression model within a Vertex AI Workbench notebook instance. Configure the model to produce feature importances post-training.",
      "B. Create an AutoML tabular regression model. Configure the model to provide explanations alongside predictions.",
      "C. Construct a custom TensorFlow neural network using Vertex AI custom training. Configure the model to provide explanations with predictions.",
      "D. Construct a random forest classification model within a Vertex AI Workbench notebook instance. Configure the model to generate feature importances post-training."
    ],
    "answer": "D. Construct a random forest classification model within a Vertex AI Workbench notebook instance. Configure the model to generate feature importances post-training."
  },
  {
    "question": "You're employed by an organization running a streaming music service with a custom production model providing \"next song\" recommendations based on user listening history. The model is deployed on a Vertex AI endpoint and recently retrained with fresh data, showing positive offline test results. Now, you aim to test the new model in production with minimal complexity. What approach should you take?",
    "options": [
      "A. Establish a new Vertex AI endpoint for the updated model and deploy it. Develop a service to randomly direct 5% of production traffic to the new endpoint. Monitor end-user metrics like listening time. Gradually increase traffic to the new endpoint if end-user metrics improve over time compared to the old model.",
      "B. Capture incoming prediction requests in BigQuery and set up an experiment in Vertex AI Experiments. Conduct batch predictions for both models using the captured data. Compare model performance side by side using the user's selected song. Deploy the new model to production if its metrics outperform the previous model.",
      "C. Deploy the new model to the existing Vertex AI endpoint and utilize traffic splitting to direct 5% of production traffic to it. Monitor end-user metrics such as listening time. If the new model outperforms the old one over time, gradually increase the traffic directed to the new model.",
      "D. Configure model monitoring for the existing Vertex AI endpoint to detect prediction drift, setting a threshold for alerts. Update the model on the endpoint from the previous one to the new model. Revert to the previous model if alerted to prediction drift."
    ],
    "answer": "C. Deploy the new model to the existing Vertex AI endpoint and utilize traffic splitting to direct 5% of production traffic to it. Monitor end-user metrics such as listening time. If the new model outperforms the old one over time, gradually increase the traffic directed to the new model."
  },
  {
    "question": "You're using Kubeflow Pipelines to build an end-to-end PyTorch-based MLOps pipeline, which involves data reading from BigQuery, processing, feature engineering, model training, evaluation, and model deployment to Cloud Storage. You're developing code for different versions of feature engineering and model training steps, running each in Vertex AI Pipelines. However, each pipeline run is taking over an hour, slowing down your development process and potentially increasing costs.\n\nWhat's the best approach to speed up execution while avoiding additional costs?",
    "options": [
      "A. Comment out the sections of the pipeline not currently being updated.",
      "B. Enable caching in all steps of the Kubeflow pipeline.",
      "C. Delegate feature engineering to BigQuery and exclude it from the pipeline.",
      "D. Add a GPU to the model training step."
    ],
    "answer": "B. Enable caching in all steps of the Kubeflow pipeline."
  },
  {
    "question": "You are employed by an international manufacturing organization that ships scientific products worldwide. These products come with instruction manuals that need translation into 15 different languages. The leadership team is interested in using machine learning to reduce the costs of manual human translations and to increase translation speed. You are tasked with implementing a scalable solution that maximizes accuracy while minimizing operational overhead. Additionally, you need to incorporate a process to evaluate and correct any incorrect translations.\n\nWhat should you do?",
    "options": [
      "A. Set up a workflow with Cloud Function triggers. Configure one Cloud Function to activate when documents are uploaded to an input Cloud Storage bucket, and another to translate these documents using the Cloud Translation API, storing the translations in an output Cloud Storage bucket. Use human reviewers to assess and correct any translation errors.",
      "B. Develop a Vertex AI pipeline that processes documents, launches an AutoML Translation training job, evaluates the translations, and deploys the model to a Vertex AI endpoint with autoscaling and model monitoring. Re-trigger the pipeline with the latest data when a predetermined skew between training and live data is detected.",
      "C. Implement AutoML Translation to train a model. Set up a Translation Hub project and use the trained model for translating documents. Employ human reviewers to evaluate and correct inaccuracies in the translations.",
      "D. Utilize Vertex AI custom training jobs to fine-tune a state-of-the-art open source pretrained model with your data. Deploy the model to a Vertex AI endpoint with autoscaling and model monitoring. Configure a trigger to initiate another training job with the latest data when a predetermined skew between training and live data is detected."
    ],
    "answer": "C. Implement AutoML Translation to train a model. Set up a Translation Hub project and use the trained model for translating documents. Employ human reviewers to evaluate and correct inaccuracies in the translations."
  },
  {
    "question": "You are developing a model to predict potential failures in a critical machine part, utilizing a dataset that includes a multivariate time series and labels indicating part failures. You have begun to experiment with various preprocessing and modeling techniques in a Vertex AI Workbench notebook.\n\nHow should you manage data logging and artifact tracking for each experiment run?",
    "options": [
      "A.\n\n1. Use the Vertex AI SDK to create an experiment and set up Vertex ML Metadata.\n\n2. Use the log_time_series_metrics function to track the preprocessed data and the log_metrics function to log loss values.",
      "B.\n\n1. Use the Vertex AI SDK to create an experiment and set up Vertex ML Metadata.\n\n2. Use the log_time_series_metrics function to track the preprocessed data, and use the log_metrics function to log loss values.",
      "C.\n\n1. Create a Vertex AI TensorBoard instance and use the Vertex AI SDK to create an experiment, associating it with the TensorBoard instance.\n\n2. Use the assign_input_artifact method to track the preprocessed data and use the log_time_series_metrics function to log loss values.",
      "D.\n\n1. Create a Vertex AI TensorBoard instance, and use the Vertex AI SDK to create an experiment, associating it with the TensorBoard instance.\n\n2. Use the log_time_series_metrics function to track the preprocessed data, and use the log_metrics function to log loss values."
    ],
    "answer": "C.\n\n1. Create a Vertex AI TensorBoard instance and use the Vertex AI SDK to create an experiment, associating it with the TensorBoard instance.\n\n2. Use the assign_input_artifact method to track the preprocessed data and use the log_time_series_metrics function to log loss values."
  },
  {
    "question": "You're developing a custom TensorFlow classification model based on tabular data stored in BigQuery. The dataset comprises hundreds of millions of rows with both categorical and numerical features. Your goal is to use a MaxMin scaler on some numerical features and apply one-hot encoding to categorical features like SKU names. The model will be trained over multiple epochs, and you aim to minimize both effort and cost.\n\nWhat approach should you take?",
    "options": [
      "A.\n\n1. Compose a SQL query to generate a separate lookup table for scaling the numerical features.\n\n2. Utilize a TensorFlow-based model from Hugging Face deployed to BigQuery to encode the text features.\n\nDirect the resulting BigQuery view into Vertex AI Training.",
      "B.\n\n1. Utilize BigQuery to scale the numerical features.\n\n2. Directly input the features into Vertex AI Training.\n\nAllow TensorFlow to handle the one-hot text encoding.",
      "C.\n\n1. Employ TFX components with Dataflow to encode the text features and scale the numerical features.\n\n2. Export the outcomes to Cloud Storage as TFRecords.\n\nInput the data into Vertex AI Training.",
      "D.\n\n1. Draft a SQL query to create a separate lookup table for scaling the numerical features.\n\n2. Perform the one-hot text encoding in BigQuery.\n\nDirect the resulting BigQuery view into Vertex AI Training."
    ],
    "answer": "D.\n\n1. Draft a SQL query to create a separate lookup table for scaling the numerical features.\n\n2. Perform the one-hot text encoding in BigQuery.\n\nDirect the resulting BigQuery view into Vertex AI Training."
  },
  {
    "question": "Your company specializes in building bridges for cities worldwide. To monitor the progress of construction projects, cameras are installed at each site. These cameras capture hourly images, which are then uploaded to a Cloud Storage bucket. A team of specialists reviews these images, selects the important ones, and annotates specific objects in them. To enhance scalability and reduce costs, you want to propose an ML solution with minimal upfront investment. What approach should you recommend?",
    "options": [
      "A. Train an AutoML object detection model to assist specialists in annotating objects in the images.",
      "B. Use the Cloud Vision API to automatically annotate objects in the images, assisting specialists with the annotation process.",
      "C. Develop a BigQuery ML classification model to identify important images and use it to help specialists filter new images.",
      "D. Utilize Vertex AI to train an open-source object detection model to assist specialists in annotating objects in the images."
    ],
    "answer": "A. Train an AutoML object detection model to assist specialists in annotating objects in the images."
  },
  {
    "question": "Your company, which specializes in selling corporate electronic products globally, has accumulated substantial historical customer data stored in BigQuery. You need to create a model to predict customer lifetime value (CLTV) over the next three years. The approach should be straightforward and efficient.\n\nWhat should you do?",
    "options": [
      "A. Create a Vertex AI Workbench notebook. Use IPython magic to run the CREATE MODEL statement to create an ARIMA model.",
      "B. Access BigQuery Studio in the Google Cloud console. Run the CREATE MODEL statement in the SQL editor to create an AutoML regression model.",
      "C. Create a Vertex AI Workbench notebook. Use IPython magic to run the CREATE MODEL statement to create an AutoML regression model.",
      "D. Access BigQuery Studio in the Google Cloud console. Run the CREATE MODEL statement in the SQL editor to create an ARIMA model."
    ],
    "answer": "B. Access BigQuery Studio in the Google Cloud console. Run the CREATE MODEL statement in the SQL editor to create an AutoML regression model."
  },
  {
    "question": "You have developed an AutoML tabular classification model to identify high-value customers engaging with your organization's website. The next step is deploying this model to a Vertex AI endpoint integrated with your website application. Since traffic is expected to increase during nights and weekends, you must configure the deployment settings to ensure low latency and cost efficiency.\n\nWhat configuration should you use?",
    "options": [
      "A. Configure the model deployment settings to use an n1-standard-32 machine type.",
      "B. Configure the model deployment settings to use an n1-standard-4 machine type. Set the minReplicaCount value to 1 and the maxReplicaCount value to 8.",
      "C. Configure the model deployment settings to use an n1-standard-4 machine type and a GPU accelerator. Set the minReplicaCount value to 1 and the maxReplicaCount value to 4.",
      "D. Configure the model deployment settings to use an n1-standard-8 machine type and a GPU accelerator."
    ],
    "answer": "B. Configure the model deployment settings to use an n1-standard-4 machine type. Set the minReplicaCount value to 1 and the maxReplicaCount value to 8."
  },
  {
    "question": "You have developed a Python module using Keras to train a regression model with two architectures: linear regression and deep neural network (DNN). The module utilizes the training_method argument to select the architecture, and for the DNN, it includes learning_rate and num_hidden_layers as hyperparameters. You plan to employ Vertex AI's hyperparameter tuning service with a budget of 100 trials to determine the optimal model architecture and hyperparameter values that minimize training loss and enhance performance.\n\nHow should you proceed?",
    "options": [
      "A. Run a single hyperparameter tuning job for 100 trials. Set num_hidden_layers as a conditional hyperparameter based on its parent hyperparameter training_method, and set learning_rate as a non-conditional hyperparameter.",
      "B. Conduct two separate hyperparameter tuning jobs: one for linear regression with 50 trials and another for DNN with 50 trials. Compare their performance on a common validation set and select the hyperparameters yielding the lowest training loss.",
      "C. Execute one hyperparameter tuning job with training_method as the hyperparameter for 50 trials. Choose the architecture with the lowest training loss, then further tune it and its corresponding hyperparameters for an additional 50 trials.",
      "D. Run a single hyperparameter tuning job for 100 trials. Set both num_hidden_layers and learning_rate as conditional hyperparameters based on their parent hyperparameter training_method."
    ],
    "answer": "D. Run a single hyperparameter tuning job for 100 trials. Set both num_hidden_layers and learning_rate as conditional hyperparameters based on their parent hyperparameter training_method."
  },
  {
    "question": "You are developing a TensorFlow Extended (TFX) pipeline with standard TFX components. The pipeline includes data preprocessing steps. After deploying the pipeline to production, it will process up to 100 TB of data stored in BigQuery. You need the data preprocessing steps to scale efficiently, publish metrics and parameters to Vertex AI Experiments, and track artifacts using Vertex ML Metadata.\n\nHow should you configure the pipeline run?",
    "options": [
      "A. Run the TFX pipeline in Vertex AI Pipelines. Configure the pipeline to use Vertex AI Training jobs with distributed processing.",
      "B. Run the TFX pipeline in Vertex AI Pipelines. Set the appropriate Apache Beam parameters in the pipeline to run the data preprocessing steps in Dataflow.",
      "C. Run the TFX pipeline in Dataproc using the Apache Beam TFX orchestrator. Set the appropriate Vertex AI permissions in the job to publish metadata in Vertex AI.",
      "D. Run the TFX pipeline in Dataflow using the Apache Beam TFX orchestrator. Set the appropriate Vertex AI permissions in the job to publish metadata in Vertex AI."
    ],
    "answer": "B. Run the TFX pipeline in Vertex AI Pipelines. Set the appropriate Apache Beam parameters in the pipeline to run the data preprocessing steps in Dataflow."
  },
  {
    "question": "You are developing a batch process to train a custom machine learning model and perform predictions. It's essential to track the lineage of both the model and the batch predictions.\n\nWhich approach should you take?",
    "options": [
      "A.\n1. Upload your dataset to BigQuery.\n2. Use a Vertex AI custom training job to train your model.\n3. Generate predictions using Vertex AI SDK custom prediction routines.",
      "B.\n1. Use Vertex AI Experiments to evaluate model performance during training.\n2. Register your model in Vertex AI Model Registry.\n3. Generate batch predictions in Vertex AI.",
      "C.\n1. Create a Vertex AI managed dataset.\n2. Use a Vertex AI training pipeline to train your model.\n3. Generate batch predictions in Vertex AI.",
      "D.\n1. Use a Vertex AI Pipelines custom training job component to train your model.\n2. Generate predictions using a Vertex AI Pipelines model batch predict component."
    ],
    "answer": "D.\n1. Use a Vertex AI Pipelines custom training job component to train your model.\n2. Generate predictions using a Vertex AI Pipelines model batch predict component."
  },
  {
    "question": "You are working at a hospital and have received approval to collect patient data for machine learning purposes. Using this data, you trained a Vertex AI tabular AutoML model to predict patient risk scores for hospital admissions. The model has been deployed, but you are concerned that over time, changes in patient demographics might alter the relationships between features, potentially affecting prediction accuracy. To address this, you need a cost-effective way to monitor for such changes and understand feature importance in the predictions.\n\nWhat should you do?",
    "options": [
      "A. Create a feature drift monitoring job. Set the sampling rate to 1 and the monitoring frequency to weekly.",
      "B. Create a feature drift monitoring job. Set the sampling rate to 0.1 and the monitoring frequency to weekly.",
      "C. Create a feature attribution drift monitoring job. Set the sampling rate to 1 and the monitoring frequency to weekly.",
      "D. Create a feature attribution drift monitoring job. Set the sampling rate to 0.1 and the monitoring frequency to weekly."
    ],
    "answer": "D. Create a feature attribution drift monitoring job. Set the sampling rate to 0.1 and the monitoring frequency to weekly."
  },
  {
    "question": "You have developed a BigQuery ML linear regression model using a training dataset stored in a BigQuery table, which receives new data every minute. To automate hourly model training and direct inference, you employ Cloud Scheduler and Vertex AI Pipelines. The feature preprocessing involves quantile bucketization and MinMax scaling on data from the past hour.\n\nTo minimize storage and computational overhead, what approach should you take?",
    "options": [
      "A. Preprocess and stage the data in BigQuery before feeding it to the model during training and inference.",
      "B. Utilize the TRANSFORM clause in the CREATE MODEL statement to compute the necessary statistics.",
      "C. Develop a component within the Vertex AI Pipelines directed acyclic graph (DAG) to calculate the required statistics and pass them to subsequent components.",
      "D. Create SQL queries to compute and store the required statistics in separate BigQuery tables, which are then referenced in the CREATE MODEL statement."
    ],
    "answer": "B. Utilize the TRANSFORM clause in the CREATE MODEL statement to compute the necessary statistics."
  }
]
questions_ep = [
    {
    "question": "You are an ML engineer at a manufacturing company. You need to build a model that identifies defects in products based on images of the product taken at the end of the assembly line. You want your model to preprocess the images with lower computation to quickly extract features of defects in products. Which approach should you use to build the model?",
    "options": [
      "A. Reinforcement learning",
      "B. Recommender system",
      "C. Recurrent Neural Networks (RNN)",
      "D. Convolutional Neural Networks (CNN)"
    ],
    "answer": "D. Convolutional Neural Networks (CNN)"
  },
  {
    "question": "You work for a large hotel chain and have been asked to assist the marketing team in gathering predictions for a targeted marketing strategy. You need to make predictions about user lifetime value (LTV) over the next 20 days so that marketing can be adjusted accordingly. The customer dataset is in BigQuery, and you are preparing the tabular data for training with AutoML Tables. This data has a time signal that is spread across multiple columns. How should you ensure that AutoML fits the best model to your data?",
    "options": [
      "A. Manually combine all columns that contain a time signal into an array. AIlow AutoML to interpret this array appropriately. Choose an automatic data split across the training, validation, and testing sets.",
      "B. Submit the data for training without performing any manual transformations. AIlow AutoML to handle the appropriate transformations. Choose an automatic data split across the training, validation, and testing sets.",
      "C. Submit the data for training without performing any manual transformations, and indicate an appropriate column as the Time column. AIlow AutoML to split your data based on the time signal provided, and reserve the more recent data for the validation and testing sets.",
      "D. Submit the data for training without performing any manual transformations. Use the columns that have a time signal to manually split your data. Ensure that the data in your validation set is from 30 days after the data in your training set and that the data in your testing sets from 30 days after your validation set."
    ],
    "answer": "D. Submit the data for training without performing any manual transformations. Use the columns that have a time signal to manually split your data. Ensure that the data in your validation set is from 30 days after the data in your training set and that the data in your testing sets from 30 days after your validation set."
  },
  {
    "question": "You have developed a BigQuery ML model that predicts customer chum, and deployed the model to Vertex AI Endpoints. You want to automate the retraining of your model by using minimal additional code when model feature values change. You also want to minimize the number of times that your model is retrained to reduce training costs. What should you do?",
    "options": [
      "A. 1 Enable request-response logging on Vertex AI Endpoints 2. Schedule a TensorFlow Data Validation job to monitor prediction drift 3. Execute model retraining if there is significant distance between the distributions",
      "B. 1. Enable request-response logging on Vertex AI Endpoints 2. Schedule a TensorFlow Data Validation job to monitor training/serving skew 3. Execute model retraining if there is significant distance between the distributions",
      "C. 1. Create a Vertex AI Model Monitoring job configured to monitor prediction drift 2. Configure alert monitoring to publish a message to a Pub/Sub queue when a monitoring alert is detected 3. Use a Cloud Function to monitor the Pub/Sub queue, and trigger retraining in BigQuery",
      "D. 1. Create a Vertex AI Model Monitoring job configured to monitor training/serving skew 2. Configure alert monitoring to publish a message to a Pub/Sub queue when a monitoring alert is detected 3. Use a Cloud Function to monitor the Pub/Sub queue, and trigger retraining in BigQuery"
    ],
    "answer": "D. 1. Create a Vertex AI Model Monitoring job configured to monitor training/serving skew 2. Configure alert monitoring to publish a message to a Pub/Sub queue when a monitoring alert is detected 3. Use a Cloud Function to monitor the Pub/Sub queue, and trigger retraining in BigQuery"
  },
  {
    "question": "You developed a Transformer model in TensorFlow to translate text. Your training data includes millions of documents in a Cloud Storage bucket. You plan to use distributed training to reduce training time. You need to configure the training job while minimizing the effort required to modify code and to manage the cluster’s configuration. What should you do?",
    "options": [
      "A. Create a Vertex AI custom training job with GPU accelerators for the second worker pool. Use tf.distribute.MultiWorkerMirroredStrategy for distribution.",
      "B. Create a Vertex AI custom distributed training job with Reduction Server. Use N1 high-memory machine type instances for the first and second pools, and use N1 high-CPU machine type instances for the third worker pool.",
      "C. Create a training job that uses Cloud TPU VMs. Use tf.distribute.TPUStrategy for distribution.",
      "D. Create a Vertex AI custom training job with a single worker pool of A2 GPU machine type instances. Use tf.distribute.MirroredStrategv for distribution."
    ],
    "answer": "A. Create a Vertex AI custom training job with GPU accelerators for the second worker pool. Use tf.distribute.MultiWorkerMirroredStrategy for distribution."
  },
  {
    "question": "You developed a Vertex AI pipeline that trains a classification model on data stored in a large BigQuery table. The pipeline has four steps, where each step is created by a Python function that uses the KubeFlow v2 API. The components have the following names:\n\nQuestion\nYou launch your Vertex AI pipeline as the following:\n\nQuestion\nYou perform many model iterations by adjusting the code and parameters of the training step. You observe high costs associated with the development, particularly the data export and preprocessing steps. You need to reduce model development costs. What should you do?",
    "options": [
      "A. Change the components’ YAML filenames to export.yaml, preprocess,yaml, f \"train- {dt}.yaml\", f\"calibrate-{dt).vaml\".",
      "B. Add the {\"kubeflow.v1.caching\": True} parameter to the set of params provided to your PipelineJob.",
      "C. Move the first step of your pipeline to a separate step, and provide a cached path to Cloud Storage as an input to the main pipeline.",
      "D. Change the name of the pipeline to f\"my-awesome-pipeline-{dt}\"."
    ],
    "answer": "B. Add the {\"kubeflow.v1.caching\": True} parameter to the set of params provided to your PipelineJob."
  },
  {
    "question": "You work for a bank. You have been asked to develop an ML model that will support loan application decisions. You need to determine which Vertex AI services to include in the workflow. You want to track the model’s training parameters and the metrics per training epoch. You plan to compare the performance of each version of the model to determine the best model based on your chosen metrics. Which Vertex AI services should you use?",
    "options": [
      "A. Vertex ML Metadata, Vertex AI Feature Store, and Vertex AI Vizier",
      "B. Vertex AI Pipelines, Vertex AI Experiments, and Vertex AI Vizier",
      "C. Vertex ML Metadata, Vertex AI Experiments, and Vertex AI TensorBoard",
      "D. Vertex AI Pipelines, Vertex AI Feature Store, and Vertex AI TensorBoard"
    ],
    "answer": "C. Vertex ML Metadata, Vertex AI Experiments, and Vertex AI TensorBoard"
  },
  {
    "question": "You are building a TensorFlow text-to-image generative model by using a dataset that contains billions of images with their respective captions. You want to create a low maintenance, automated workflow that reads the data from a Cloud Storage bucket collects statistics, splits the dataset into training/validation/test datasets performs data transformations trains the model using the training/validation datasets, and validates the model by using the test dataset. What should you do?",
    "options": [
      "A. Use the Apache Airflow SDK to create multiple operators that use Dataflow and Vertex AI services. Deploy the workflow on Cloud Composer.",
      "B. Use the MLFlow SDK and deploy it on a Google Kubernetes Engine cluster. Create multiple components that use Dataflow and Vertex AI services.",
      "C. Use the Kubeflow Pipelines (KFP) SDK to create multiple components that use Dataflow and Vertex AI services. Deploy the workflow on Vertex AI Pipelines.",
      "D. Use the TensorFlow Extended (TFX) SDK to create multiple components that use Dataflow and Vertex AI services. Deploy the workflow on Vertex AI Pipelines."
    ],
    "answer": "D. Use the TensorFlow Extended (TFX) SDK to create multiple components that use Dataflow and Vertex AI services. Deploy the workflow on Vertex AI Pipelines."
  },
  {
    "question": "Your team frequently creates new ML models and runs experiments. Your team pushes code to a single repository hosted on Cloud Source Repositories. You want to create a continuous integration pipeline that automatically retrains the models whenever there is any modification of the code. What should be your first step to set up the CI pipeline?",
    "options": [
      "A. Configure a Cloud Build trigger with the event set as \"Pull Request\"",
      "B. Configure a Cloud Build trigger with the event set as \"Push to a branch\"",
      "C. Configure a Cloud Function that builds the repository each time there is a code change",
      "D. Configure a Cloud Function that builds the repository each time a new branch is created"
    ],
    "answer": "B. Configure a Cloud Build trigger with the event set as \"Push to a branch\""
  },
  {
    "question": "You need to use TensorFlow to train an image classification model. Your dataset is located in a Cloud Storage directory and contains millions of labeled images. Before training the model, you need to prepare the data. You want the data preprocessing and model training workflow to be as efficient, scalable, and low maintenance as possible. What should you do?",
    "options": [
      "A. 1. Create a Dataflow job that creates sharded TFRecord files in a Cloud Storage directory. 2. Reference tf.data.TFRecordDataset in the training script. 3. Train the model by using Vertex AI Training with a V100 GPU.",
      "B. 1. Create a Dataflow job that moves the images into multiple Cloud Storage directories, where each directory is named according to the corresponding label 2. Reference tfds.folder_dataset:ImageFolder in the training script. 3. Train the model by using Vertex AI Training with a V100 GPU.",
      "C. 1. Create a Jupyter notebook that uses an nt-standard-64 V100 GPU Vertex AI Workbench instance. 2. Write a Python script that creates sharded TFRecord files in a directory inside the instance. 3. Reference tf.data.TFRecordDataset in the training script. 4. Train the model by using the Workbench instance.",
      "D. 1. Create a Jupyter notebook that uses an n1-standard-64, V100 GPU Vertex AI Workbench instance. 2. Write a Python script that copies the images into multiple Cloud Storage directories, where each. directory is named according to the corresponding label. 3. Reference tfds.foladr_dataset.ImageFolder in the training script. 4. Train the model by using the Workbench instance."
    ],
    "answer": "A. 1. Create a Dataflow job that creates sharded TFRecord files in a Cloud Storage directory. 2. Reference tf.data.TFRecordDataset in the training script. 3. Train the model by using Vertex AI Training with a V100 GPU."
  },
  {
    "question": "You are developing a model to identify traffic signs in images extracted from videos taken from the dashboard of a vehicle. You have a dataset of 100,000 images that were cropped to show one out of ten different traffic signs. The images have been labeled accordingly for model training, and are stored in a Cloud Storage bucket. You need to be able to tune the model during each training run. How should you train the model?",
    "options": [
      "A. Train a model for object detection by using Vertex AI AutoML.",
      "B. Train a model for image classification by using Vertex AI AutoML.",
      "C. Develop the model training code for object detection, and train a model by using Vertex AI custom training.",
      "D. Develop the model training code for image classification, and train a model by using Vertex AI custom training."
    ],
    "answer": "D. Develop the model training code for image classification, and train a model by using Vertex AI custom training."
  },
  {
    "question": "Your team is training a large number of ML models that use different algorithms, parameters, and datasets. Some models are trained in Vertex AI Pipelines, and some are trained on Vertex AI Workbench notebook instances. Your team wants to compare the performance of the models across both services. You want to minimize the effort required to store the parameters and metrics. What should you do?",
    "options": [
      "A. Implement an additional step for all the models running in pipelines and notebooks to export parameters and metrics to BigQuery.",
      "B. Create a Vertex AI experiment. Submit all the pipelines as experiment runs. For models trained on notebooks log parameters and metrics by using the Vertex AI SDK.",
      "C. Implement all models in Vertex AI Pipelines Create a Vertex AI experiment, and associate all pipeline runs with that experiment.",
      "D. Store all model parameters and metrics as model metadata by using the Vertex AI Metadata API."
    ],
    "answer": "B. Create a Vertex AI experiment. Submit all the pipelines as experiment runs. For models trained on notebooks log parameters and metrics by using the Vertex AI SDK."
  },
  {
    "question": "You built a deep learning-based image classification model by using on-premises data. You want to use Vertex AI to deploy the model to production. Due to security concerns, you cannot move your data to the cloud. You are aware that the input data distribution might change over time. You need to detect model performance changes in production. What should you do?",
    "options": [
      "A. Use Vertex Explainable AI for model explainability. Configure feature-based explanations.",
      "B. Use Vertex Explainable AI for model explainability. Configure example-based explanations.",
      "C. Create a Vertex AI Model Monitoring job. Enable training-serving skew detection for your model.",
      "D. Create a Vertex AI Model Monitoring job. Enable feature attribution skew and drift detection for your model."
    ],
    "answer": "D. Create a Vertex AI Model Monitoring job. Enable feature attribution skew and drift detection for your model."
  },
  {
    "question": "You are working on a prototype of a text classification model in a managed Vertex AI Workbench notebook. You want to quickly experiment with tokenizing text by using a Natural Language Toolkit (NLTK) library. How should you add the library to your Jupyter kernel?",
    "options": [
      "A. Install the NLTK library from a terminal by using the pip install nltk command.",
      "B. Write a custom Dataflow job that uses NLTK to tokenize your text and saves the output to Cloud Storage.",
      "C. Create a new Vertex AI Workbench notebook with a custom image that includes the NLTK library.",
      "D. Install the NLTK library from a Jupyter cell by using the !pip install nltk --user command."
    ],
    "answer": "D. Install the NLTK library from a Jupyter cell by using the !pip install nltk --user command."
  },
  {
    "question": "You have developed an application that uses a chain of multiple scikit-learn models to predict the optimal price for your company’s products. The workflow logic is shown in the diagram. Members of your team use the individual models in other solution workflows. You want to deploy this workflow while ensuring version control for each individual model and the overall workflow. Your application needs to be able to scale down to zero. You want to minimize the compute resource utilization and the manual effort required to manage this solution. What should you do?",
    "options": [
      "A. Expose each individual model as an endpoint in Vertex AI Endpoints. Create a custom container endpoint to orchestrate the workflow.",
      "B. Create a custom container endpoint for the workflow that loads each model’s individual files Track the versions of each individual model in BigQuery.",
      "C. Expose each individual model as an endpoint in Vertex AI Endpoints. Use Cloud Run to orchestrate the workflow.",
      "D. Load each model’s individual files into Cloud Run. Use Cloud Run to orchestrate the workflow. Track the versions of each individual model in BigQuery."
    ],
    "answer": "C. Expose each individual model as an endpoint in Vertex AI Endpoints. Use Cloud Run to orchestrate the workflow."
  },
  {
    "question": "You work for a semiconductor manufacturing company. You need to create a real-time application that automates the quality control process. High-definition images of each semiconductor are taken at the end of the assembly line in real time. The photos are uploaded to a Cloud Storage bucket along with tabular data that includes each semiconductor’s batch number, serial number, dimensions, and weight. You need to configure model training and serving while maximizing model accuracy. What should you do?",
    "options": [
      "A. Use Vertex AI Data Labeling Service to label the images, and tram an AutoML image classification model. Deploy the model, and configure Pub/Sub to publish a message when an image is categorized into the failing class.",
      "B. Use Vertex AI Data Labeling Service to label the images, and train an AutoML image classification model. Schedule a daily batch prediction job that publishes a Pub/Sub message when the job completes.",
      "C. Convert the images into an embedding representation. Import this data into BigQuery, and train a BigQuery ML K-means clustering model with two clusters. Deploy the model and configure Pub/Sub to publish a message when a semiconductor’s data is categorized into the failing cluster.",
      "D. Import the tabular data into BigQuery, use Vertex AI Data Labeling Service to label the data and train an AutoML tabular classification model. Deploy the model, and configure Pub/Sub to publish a message when a semiconductor’s data is categorized into the failing class."
    ],
    "answer": "A. Use Vertex AI Data Labeling Service to label the images, and tram an AutoML image classification model. Deploy the model, and configure Pub/Sub to publish a message when an image is categorized into the failing class."
  },
  {
    "question": "You work at a large organization that recently decided to move their ML and data workloads to Google Cloud. The data engineering team has exported the structured data to a Cloud Storage bucket in Avro format. You need to propose a workflow that performs analytics, creates features, and hosts the features that your ML models use for online prediction. How should you configure the pipeline?",
    "options": [
      "A. Ingest the Avro files into Cloud Spanner to perform analytics. Use a Dataflow pipeline to create the features, and store them in Vertex AI Feature Store for online prediction.",
      "B. Ingest the Avro files into BigQuery to perform analytics. Use a Dataflow pipeline to create the features, and store them in Vertex AI Feature Store for online prediction.",
      "C. Ingest the Avro files into Cloud Spanner to perform analytics. Use a Dataflow pipeline to create the features, and store them in BigQuery for online prediction.",
      "D. Ingest the Avro files into BigQuery to perform analytics. Use BigQuery SQL to create features and store them in a separate BigQuery table for online prediction."
    ],
    "answer": "B. Ingest the Avro files into BigQuery to perform analytics. Use a Dataflow pipeline to create the features, and store them in Vertex AI Feature Store for online prediction."
  },
  {
    "question": "You work for a multinational organization that has recently begun operations in Spain. Teams within your organization will need to work with various Spanish documents, such as business, legal, and financial documents. You want to use machine learning to help your organization get accurate translations quickly and with the least effort. Your organization does not require domain-specific terms or jargon. What should you do?",
    "options": [
      "A. Create a Vertex AI Workbench notebook instance. In the notebook, extract sentences from the documents, and train a custom AutoML text model.",
      "B. Use Google Translate to translate 1,000 phrases from Spanish to English. Using these translated pairs, train a custom AutoML Translation model.",
      "C. Use the Document Translation feature of the Cloud Translation API to translate the documents.",
      "D. Create a Vertex AI Workbench notebook instance. In the notebook, convert the Spanish documents into plain text, and create a custom TensorFlow seq2seq translation model."
    ],
    "answer": "C. Use the Document Translation feature of the Cloud Translation API to translate the documents."
  },
  {
    "question": "You have a custom job that runs on Vertex AI on a weekly basis. The job is implemented using a proprietary ML workflow that produces the datasets, models, and custom artifacts, and sends them to a Cloud Storage bucket. Many different versions of the datasets and models were created. Due to compliance requirements, your company needs to track which model was used for making a particular prediction, and needs access to the artifacts for each model. How should you configure your workflows to meet these requirements?",
    "options": [
      "A. Use the Vertex AI Metadata API inside the custom job to create context, execution, and artifacts for each model, and use events to link them together.",
      "B. Create a Vertex AI experiment, and enable autologging inside the custom job.",
      "C. Configure a TensorFlow Extended (TFX) ML Metadata database, and use the ML Metadata API.",
      "D. Register each model in Vertex AI Model Registry, and use model labels to store the related dataset and model information."
    ],
    "answer": "A. Use the Vertex AI Metadata API inside the custom job to create context, execution, and artifacts for each model, and use events to link them together."
  },
  {
    "question": "You have recently developed a custom model for image classification by using a neural network. You need to automatically identify the values for learning rate, number of layers, and kernel size. To do this, you plan to run multiple jobs in parallel to identify the parameters that optimize performance. You want to minimize custom code development and infrastructure management. What should you do?",
    "options": [
      "A. Train an AutoML image classification model.",
      "B. Create a custom training job that uses the Vertex AI Vizier SDK for parameter optimization.",
      "C. Create a Vertex AI hyperparameter tuning job.",
      "D. Create a Vertex AI pipeline that runs different model training jobs in parallel."
    ],
    "answer": "C. Create a Vertex AI hyperparameter tuning job."
  },
  {
    "question": "You are tasked with building an MLOps pipeline to retrain tree-based models in production. The pipeline will include components related to data ingestion, data processing, model training, model evaluation, and model deployment. Your organization primarily uses PySpark-based workloads for data preprocessing. You want to minimize infrastructure management effort. How should you set up the pipeline?",
    "options": [
      "A. Set up a TensorFlow Extended (TFX) pipeline on Vertex AI Pipelines to orchestrate the MLOps pipeline. Write a custom component for the PySpark-based workloads on Dataproc.",
      "B. Set up a Vertex AI Pipelines to orchestrate the MLOps pipeline. Use the predefined Dataproc component for the PySpark-based workloads.",
      "C. Set up Kubeflow Pipelines on Google Kubernetes Engine to orchestrate the MLOps pipeline. Write a custom component for the PySparkbased workloads on Dataproc.",
      "D. Set up Cloud Composer to orchestrate the MLOps pipeline. Use Dataproc workflow templates for the PySpark-based workloads in Cloud Composer."
    ],
    "answer": "B. Set up a Vertex AI Pipelines to orchestrate the MLOps pipeline. Use the predefined Dataproc component for the PySpark-based workloads."
  }
]
with tab1:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

   
    # Function to create a table of contents
    def create_toc(questions_t1):
        toc = ""
        for idx, q in enumerate(questions_t1):
            toc += f"- [Question {idx + 1}](#{'t1-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_t1)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 1")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_t1):
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


    # Function to create a table of contents
    def create_toc(questions_t2):
        toc = ""
        for idx, q in enumerate(questions_t2):
            toc += f"- [Question {idx + 1}](#{'t2-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_t2)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 2")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_t2):
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

   
    # Function to create a table of contents
    def create_toc(questions_t3):
        toc = ""
        for idx, q in enumerate(questions_t3):
            toc += f"- [Question {idx + 1}](#{'t3-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_t3)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 3")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_t3):
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

with tab4:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

   
    # Function to create a table of contents
    def create_toc(questions_t4):
        toc = ""
        for idx, q in enumerate(questions_t4):
            toc += f"- [Question {idx + 1}](#{'t4-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_t4)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 4")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_t4):
        st.subheader(f"Question {idx + 1}", anchor=f"t4-question-{idx+1}")

        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"t4_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

with tab5:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

   
    # Function to create a table of contents
    def create_toc(questions_t5):
        toc = ""
        for idx, q in enumerate(questions_t5):
            toc += f"- [Question {idx + 1}](#{'t5-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_t5)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 5")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_t5):
        st.subheader(f"Question {idx + 1}", anchor=f"t5-question-{idx+1}")

        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"t5_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

with tab6:
    st.title("GCP MLE Practice Questions")
    "Material from [Alex Levkovich's course on Udemy](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-exams/)"

   
    # Function to create a table of contents
    def create_toc(questions_t6):
        toc = ""
        for idx, q in enumerate(questions_t6):
            toc += f"- [Question {idx + 1}](#{'t6-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_t6)

    # Sidebar with table of contents
    st.sidebar.markdown("## Test 6")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_t6):
        st.subheader(f"Question {idx + 1}", anchor=f"t6-question-{idx+1}")

        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"t6_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

with tab_ep:
    st.title("GCP MLE Practice Questions")
    "Additional questions from [Examprepper](https://www.examprepper.co/)"

   
    # Function to create a table of contents
    def create_toc(questions_ep):
        toc = ""
        for idx, q in enumerate(questions_ep):
            toc += f"- [Question {idx + 1}](#{'ep-question-' + str(idx + 1)})\n"
        return toc

    # Create the table of contents
    toc = create_toc(questions_ep)

    # Sidebar with table of contents
    st.sidebar.markdown("## Examprepper")
    st.sidebar.markdown(toc)

    # Loop through the questions
    for idx, q in enumerate(questions_ep):
        st.subheader(f"Question {idx + 1}", anchor=f"ep-question-{idx+1}")

        # Image subline
        if idx+1==5:
            st.image('https://img.examtopics.com/professional-machine-learning-engineer/image3.png')
            st.image('https://img.examtopics.com/professional-machine-learning-engineer/image4.png')
        elif idx+1==14:
            st.image('https://img.examtopics.com/professional-machine-learning-engineer/image5.png')

        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"ep_q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")

with tab_rand:
    random_qn = st.button('Try a random question')
    if random_qn:
        # Pick a random question and store it in session state
        st.session_state.current_question = secrets.choice(questions_t1+questions_t2+questions_t3+questions_t4+questions_t5+questions_t6)

    if st.session_state.current_question:
        q = st.session_state.current_question
        st.write(q['question'])
        user_answer = st.radio("a", q["options"], index=None, key=f"q{idx}",
                            label_visibility='hidden')

        if user_answer:
            if user_answer in q["answer"]:
                st.success("Correct!")
            else:
                st.error("Incorrect. Try again!")
