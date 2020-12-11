TAG=pdi-mdp
PWD:=$(shell pwd)
WORK_PATH=/home/jovyan/work

all: build run

build:
	docker build -t $(TAG) .

run:
	docker run \
		-v "$(PWD):$(WORK_PATH)" \
		-p 8888:8888 $(TAG)
