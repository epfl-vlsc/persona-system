#!/usr/bin/env python3

import argparse
import os
import subprocess
import shlex
import sys

script_dir = os.path.dirname(__file__)

def get_args():
    parser = argparse.ArgumentParser(description="Build a given python image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--docker-org", default="epflpersona", help="the persona org to make this image for")
    parser.add_argument("-f", "--file", help="if specified, override the default Dockerfile that this script guesses")
    parser.add_argument("-p", "--push", default=False, action="store_true", help="push the new image to dockerhub")
    parser.add_argument("name", help="the type of image to create")
    args = parser.parse_args()
    if args.file is None:
        fl = os.path.join(script_dir, "persona_{}.docker".format(args.name))
        if os.path.exists(fl):
            args.file = fl
        else:
            parser.error("Presumed docker file {} not found. Maybe specify a file?".format(fl))
    return args

def run(args):
    docker_name = "/".join((args.docker_org, args.name))
    subprocess.run(shlex.split("docker build -t {docker_name} -f {dockerfile} {rundir}".format(
        docker_name=docker_name, dockerfile=args.file, rundir=script_dir
    )), check=True, stdout=sys.stdout, stderr=sys.stderr)
    if args.push:
        subprocess.run(shlex.split("docker push {docker_name}".format(docker_name=docker_name)))

if __name__ == "__main__":
    args = get_args()
    run(args=args)
