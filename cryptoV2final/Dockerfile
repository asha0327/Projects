# Use Python 3.10.12 as the base image
FROM python:3.9.13

# Set the working directory to /code
WORKDIR /code

# Copy the requirements file into the container at /code/requirements.txt
COPY ./requirements.txt /code/requirements.txt

# Upgrade pip and install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory for the application
WORKDIR $HOME/app

# Copy the local code into the container at /home/user/app
COPY --chown=user . $HOME/app

# Specify the command to run on container start
CMD ["streamlit", "run", "app.py","--server.port","7860"]
