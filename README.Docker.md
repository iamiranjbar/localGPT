### Building and running your application

! For make available to read models from container, comment and uncomment first lines of settings.py.

First, build your image with this command : 

`docker build -t my-chatbot-app .`

Then, run the built image by this command:

`docker run -p 8501:8501 -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models -v /Users/amirranjbar/Desktop/Military/Code/tokenizers/local_sbert_model:/app/tokenizers/local_sbert_model -v /Users/amirranjbar/Desktop/Military/Code/localDocs:/app/localDocs my-chatbot-app`

- With v tag, we mount models path into the container to access them and keep image size small.

! Building server for chatbot and serve APIs, comment and uncomment the last lines of Dockerfile and run the build command with different tag name, e.g.:

`docker build -t chat-server .`

Then, run the built image by this command:

`docker run -p 8000:8000 -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models -v /Users/amirranjbar/Desktop/Military/Code/tokenizers/local_sbert_model:/app/tokenizers/local_sbert_model -v /Users/amirranjbar/Desktop/Military/Code/localDocs:/app/localDocs chat-server`

There is another way to provide both of them, we can comment out the last two lines of Dockerfile. Then build the base image with the below command:

`docker build -t localgpt .`

Then, run the built image by this command in the background:

`docker run -d -p 8000:8000 -p 8501:8501 --name localgpt -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models -v /Users/amirranjbar/Desktop/Military/Code/tokenizers/local_sbert_model:/app/tokenizers/local_sbert_model -v /Users/amirranjbar/Desktop/Military/Code/localDocs:/app/localDocs localgpt`

Then, you can run commands inside your container with below commands:

`docker exec localgpt streamlit run src/app.py `
`docker exec localgpt python src/server.py `

Also you can get an interactive bash to interact with container with the below command:

`docker exec -it localgpt /bin/bash`

### Deploying your application to the cloud

If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.
Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.


## Persisting the database
To ensure that your database persists even if the container is stopped or restarted, you can add a volume for the database. This can be done by modifying the docker run command to include a volume mount for the db directory.

Add the following volume mount to your docker run command:

`-v /path/on/host/db:/app/db`

For example, your docker run command for the Streamlit app would look like this:

`docker run -p 8501:8501 -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models -v /Users/amirranjbar/Desktop/Military/Code/tokenizers/local_sbert_model:/app/tokenizers/local_sbert_model -v /Users/amirranjbar/Desktop/Military/Code/localDocs:/app/localDocs -v /path/on/host/db:/app/db my-chatbot-app`

This command includes the volume mount for the database persistence. Replace `/path/on/host/db` with the actual path on your host machine where you want to store the database files.

Similarly, for the server version, you would use:

`docker run -p 8000:8000 -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models -v /Users/amirranjbar/Desktop/Military/Code/tokenizers/local_sbert_model:/app/tokenizers/local_sbert_model -v /Users/amirranjbar/Desktop/Military/Code/localDocs:/app/localDocs -v /path/on/host/db:/app/db chat-server`

And for the combined version:

`docker run -d -p 8000:8000 -p 8501:8501 --name localgpt -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models -v /Users/amirranjbar/Desktop/Military/Code/tokenizers/local_sbert_model:/app/tokenizers/local_sbert_model -v /Users/amirranjbar/Desktop/Military/Code/localDocs:/app/localDocs -v /path/on/host/db:/app/db localgpt`

Remember to replace `/path/on/host/db` with your desired host path in all these commands.
