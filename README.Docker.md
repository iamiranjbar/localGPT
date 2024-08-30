### Building and running your application

! For make available to read models from container, comment and uncomment first lines of settings.py.

First, build your image with this command : 

`docker build -t my-chatbot-app .`

Then, run the built image by this command:

`docker run -p 8501:8501 -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models my-chatbot-app`

- With v tag, we mount models path into the container to access them and keep image size small.

! Building server for chatbot and serve APIs, comment and uncomment the last lines of Dockerfile and run the build command with different tag name, e.g.:

`docker build -t chat-server .`

Then, run the built image by this command:

`docker run -p 8000:8000 -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models my-chatbot-app`

There is another way to provide both of them, we can comment out the last two lines of Dockerfile. Then build the base image with the below command:

`docker build -t localgpt .`

Then, run the built image by this command in the background:

`docker run -d -p 8000:8000 -p 8501:8501 --name localgpt -v /Users/amirranjbar/Desktop/Military/Code/models:/app/models localgpt`

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




