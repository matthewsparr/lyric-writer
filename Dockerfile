FROM heroku/miniconda

# Grab requirements.txt.
ADD requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt
RUN conda install -c conda-forge cxx-compiler 
RUN conda install theano
