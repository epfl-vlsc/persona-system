#!/usr/bin/env python3

import argparse
import os
import subprocess
import shlex
import sys
import itertools
import tempfile

script_dir = os.path.dirname(__file__)
project_root = os.path.join(script_dir, "/".join(itertools.repeat("..", 5)))

def get_args():
    parser = argparse.ArgumentParser(description="Build the full python image, with our package installed",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--docker-org", help="the docker dub org for which to make this image. defaults to private local image")
    parser.add_argument("-f", "--file", default=os.path.join(script_dir, "persona_full.docker"), help="if specified, override the default Dockerfile")
    parser.add_argument("-i", "--image", default="epflpersona/build", help="the docker image to use for building the pip package")
    parser.add_argument("-n", "--name", default="persona_full", help="the name of this image, used to tag")
    args = parser.parse_args()
    if not (os.path.exists(args.file) and os.path.isfile(args.file)):
        parser.error("docker file {} doesn't exist or isn't a file".format(args.file))
    return args

def build_pip_package(pip_dir, docker_image):
    run_script = """docker run --rm --entrypoint tensorflow/tools/build_pip_pkg.sh -v {tf_dir}:/workspace -v {pip_dir}:/pip_dir -w /workspace {img_name} /pip_dir """.format(
            tf_dir=os.path.abspath(project_root), pip_dir=os.path.abspath(pip_dir), img_name=docker_image
        )
    subprocess.run(shlex.split(run_script), check=True, stdout=sys.stdout, stderr=sys.stderr)

def install_pip_package(pip_dir):
    wheel_file = [f for f in os.listdir(pip_dir) if f.endswith(".whl")]
    assert len(wheel_file) == 1
    wheel_file = os.path.join(pip_dir, wheel_file[0])
    if args.docker_org is None:
        name = args.name
    else:
        name = "/".join((args.org, args.name))
    subprocess.run(shlex.split(
        "docker build --build-arg pip_source={wheel_file} -t {name} -f {docker_file} {run_dir}".format(
            wheel_file=wheel_file, name=name, docker_file=args.file, run_dir=script_dir
        )
    ), check=True, stdout=sys.stdout, stderr=sys.stderr)

def run(args):
    with tempfile.TemporaryDirectory(dir=script_dir) as pip_dir:
        build_pip_package(pip_dir=pip_dir, docker_image=args.image)
        install_pip_package(pip_dir=pip_dir)

if __name__ == "__main__":
    run(get_args())
