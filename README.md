# SleepRecorder

## Server

git clone https://github.com/TilioChr/SleepRecorder.git
cd sleep-recorder
cp .env.example .env
./deploy.sh

## RaspberryPi

arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 | nc <IP_SERVER> 5000
