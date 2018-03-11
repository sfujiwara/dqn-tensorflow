#!/usr/bin/env bash

#!/usr/bin/env bash

# Update an upgrade
apt-get update
apt-get -y upgrade

# Install google-fluentd
curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
sha256sum install-logging-agent.sh
sudo bash install-logging-agent.sh

# Create config file for google-fluentd
FLUENTD_CONF_FILE="/etc/google-fluentd/config.d/python.conf"
echo "<source>" > ${FLUENTD_CONF_FILE}
echo "  type tail" >> ${FLUENTD_CONF_FILE}
echo "  format json" >> ${FLUENTD_CONF_FILE}
echo "  path /var/log/python/*.log,/var/log/python/*.json" >> ${FLUENTD_CONF_FILE}
echo "  read_from_head true" >> ${FLUENTD_CONF_FILE}
echo "  tag python" >> ${FLUENTD_CONF_FILE}
echo "</source>" >> ${FLUENTD_CONF_FILE}

# Create log directory for Python script
mkdir -p /var/log/python

# Restart google-fluentd
service google-fluentd restart

# Install python modules
apt-get -y install python-pip
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
apt-get install -y emacs

pip install -U pip
pip install numpy
pip install pandas
pip install ortools
pip install gym[atari]
pip install ipython
pip install tensorflow
pip install scipy
pip isntall scikit-image
pip install tqdm
pip install git+https://github.com/sfujiwara/gjhandler.git
