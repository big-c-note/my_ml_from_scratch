FROM python:3.7
# Make directory.
RUN mkdir /my_ml
# Work from the root of the repo.
WORKDIR /my_ml
# Copy the requirements.
COPY requirements.txt requirements.txt 
# Install python modules.
RUN pip3 install -r requirements.txt
# Copy everything else.
COPY . /my_ml
RUN pip install -e .
# Setup tests.
CMD ["/bin/bash"]
