#!/usr/bin/env bash
set -eu
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_IMAGE_NAME="epflpersona/persona_build"
DOCKERFILE="persona_build.docker"

docker build -t "${DOCKER_IMAGE_NAME}" -f "${SCRIPT_DIR}/${DOCKERFILE}" "${SCRIPT_DIR}"
