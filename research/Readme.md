## Setup

### Docker Setup
The agent under evaluation is given the power to make tool calls to execute python scripts and upload excel/csv files on a docker container. To run eval, you must setup the `tab-div-code-sandbox:latest` docker image on your system as follows:
1. Start the docker service in your system following these instructions([Windows](https://stackoverflow.com/a/44182489), [Unix/Linux](https://docs.docker.com/engine/daemon/start/)).
2. Navigate to `research/agents` directory and run `docker build -t tab-div-code-sandbox:latest .` to build the `tab-div-code-sandbox` docker image