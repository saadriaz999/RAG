# Rag Based Medical Question Answer Dataset


## Create a virtual environment

Create the environment:

`python -m venv .venv`


Activate the virtual environment:

`.venv\Scripts\activate`


## Setup Docker

Pull the docker image:

`docker pull qdrant/qdrant`

Run the docker container:

`docker run -p 6333:6333 --name database qdrant/qdrant`

