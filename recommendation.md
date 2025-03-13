# Dynamic Website to let SWE folks know whether they are underpaid for the Experience and skills using ML and deploy it on AWS
Deploying a Dynamic Website for a Café on AWS
Introduction
Deploying a dynamic website for a café on AWS involves several steps, including reviewing existing configurations and resources, configuring the EC2 instance using Cloud9, installing the café application, and deploying the website.

Step One: Reviewing the Existing Configurations and Resources
The existing configurations and resources include a public IP address, a security group that only allows SSH access on port 22, and a Cloud9 environment connected to the EC2 instance. The security group does not allow HTTP access, which needs to be fixed soon.

Step Two: Configuring the EC2 Instance Using Cloud9
To configure the EC2 instance using Cloud9, you need to run commands to observe the operating system, web server, database, and PHP details. You also need to start the webserver and database and set them to start automatically on future instance reboots for persistence.

Step Three: Installing the Café Application
To install the café application, you need to run wget commands to download and extract the web server application files, which were saved on S3. You then need to unzip the cafe.zip file and move its content to the HTML directory.

Deploying a Dynamic Website for a Café on AWS
To deploy a dynamic website for a café on AWS, you can use the following steps:

Create an S3 bucket to host the static website.
Create an API Gateway to handle the dynamic content.
Create a Lambda function to process the dynamic content.
Create a DynamoDB table to store the dynamic data.
Configure the API Gateway to call the Lambda function.
Configure the Lambda function to read from the DynamoDB table.
Using Machine Learning to Determine Underpayment
To determine whether a software engineer (SWE) is underpaid for their experience and skills using machine learning, you can use the following steps:

Collect data on SWE salaries, experience, and skills.
Train a machine learning model to predict SWE salaries based on experience and skills.
Use the trained model to predict the salary for a given SWE.
Compare the predicted salary to the actual salary to determine underpayment.
Deploying the Website on AWS
To deploy the website on AWS, you can use the following steps:

Create an S3 bucket to host the static website.
Create an API Gateway to handle the dynamic content.
Create a Lambda function to process the dynamic content.
Create a DynamoDB table to store the dynamic data.
Configure the API Gateway to call the Lambda function.
Configure the Lambda function to read from the DynamoDB table.
Conclusion
Deploying a dynamic website for a café on AWS involves several steps, including reviewing existing configurations and resources, configuring the EC2 instance using Cloud9, installing the café application, and deploying the website. Using machine learning to determine underpayment for SWEs involves collecting data, training a machine learning model, and deploying the website on AWS.

References
tarek-ismail.medium.com
aws.amazon.com